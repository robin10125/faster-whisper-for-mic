import pyaudio
import io
import wave
import multiprocessing
from faster_whisper_mic.faster_whisper_mic import WhisperModel
#from faster_whisper_mic import WhisperModel
import datetime

# TODO:
#       Set up algorithm to chunk audio and text based off of periods of silence
#       Implement silence detection
#       Use voice detection to have adaptable latency for model prediction batches
###############################################################################################

###--- Audio recording parameters ---###
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

AUDIO_CHUNK_LENGTH = 4 # Length of audio chunks in seconds
device_id = 6
###--- End Audio recording parameters ---###

###############################################################################################

######------ Functions ------######

#- Multiprocessing functions -#

#TODO: Implement verified/immutable transcript vs unverified transcript, so that verified transcript is not changed or taking up compuation resources
def model_server(input_queue, output_queue, buffer_prune_queue,):
    """
    model_server transcribes audio data into text segments.

    Args:
        input_queue (Queue): Queue for receiving audio data from the recording program.
        output_queue (Queue): Queue for sending transcribed text segments to the display process.
        buffer_prune_queue (Queue): Queue for sending pruning location in the audio buffer.

    Returns:
        None
    """

    model = initialize_model()
    
    confirmed_transcript=''
    while True:
        # Receive audio data from the recording program
        audio_data = input_queue.get()

        # NOTE: Temporary shutdown signal
        if audio_data is None: 
            print("\nTranscription process terminated.") 
            break
        
        time_at = datetime.datetime.now().strftime("%H:%M:%S")
        unconfirmed_transcript=''
        
        # Transcribe the audio data into segments of text
        vad_audio, speech_chunks = model.vad_filter(audio=audio_data)
        print(f'speech chunks: {speech_chunks}')
        segments, info = model.transcribe(vad_audio, speech_chunks=speech_chunks, beam_size=5, word_timestamps=True)

        num_sentences = 0
        words_list = []

        # segments is a generator object that will use the whisper model autoregressively to generate text transcripts from the audio data provided in model.transcribe
        for segment in segments:
            for word in segment.words:
                if "." in word.word or "?" in word.word or "!" in word.word or "..." in word.word:
                    num_sentences += 1
                    words_list.append((word.start, word.end, word.word, True))
                else:
                    words_list.append((word.start, word.end, word.word, False))
        
        # confirmed_signal tells the program where to prune the audio buffer
        confirmed_signal = 0
        
        print(f'num_sentences: {num_sentences}')
        
        # prune all but the last full sentence and all words afterwards, and calculate the timestamp in the audio buffer of this pruning location
        if num_sentences == 2 and words_list[-1][3] == False or num_sentences > 2:
            prune_prime = False
            # Backward loop through words_list to find the index of the end of the second last full sentence
            for i in range(len(words_list)-1, -1, -1):
                word_tuple = words_list[i]
                if word_tuple[3] == True: 
                    if prune_prime == True:
                        confirmed_signal = i
                        buffer_prune_queue.put(word_tuple[1])
                        break
                    prune_prime = True

            # Forward loop through words_list to create the confirmed transcript        
            for j in range(confirmed_signal+1):
                confirmed_transcript += words_list[j][2]

        # Forward loop through words_list to create the unconfirmed transcript (executes regardless of num sentences)
        for k in range(confirmed_signal, len(words_list)):
            unconfirmed_transcript += words_list[k][2]
        
        output = (str(time_at), confirmed_transcript, unconfirmed_transcript)
        output_queue.put(output)

def initialize_model(size="tiny.en"):
    """
    Initializes a WhisperModel object with the specified size.

    Args:
        size (str, optional): The size of the model. Defaults to "tiny.en".

    Returns:
        WhisperModel: The initialized WhisperModel object.
    """

    model_size = size
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    return model


def output_transcript(output_queue):
    """
    Writes the transcripts to a file.

    Args:
        output_queue (Queue): The queue containing the text data.

    Returns:
        None
    """
    
    while True:
        text_data = output_queue.get()
        
        #confirmed_transcript = text_data[1]
        #unconfirmed_transcript = text_data[2]

        print(f'Confirmed transcript: \n {text_data[1]} \n' 
            f'Unconfirmed transcript: \n {text_data[2]} \n Last Updated: {text_data[0]} \n'
            f'############################################################################################################################# \n')


#- End Multiprocessing functions -#

###############################################################################################

#- Audio processing functions -#

def process_audio_chunk(frames, sample_width):
    """
    Process audio frames and convert them into a wav file-like BytesIO object.

    Args:
        frames (list): List of audio frames.
        sample_width (int): Width of each audio sample in bytes.

    Returns:
        BytesIO: Wav file-like BytesIO object containing the processed audio.

    """
    wav_stream = io.BytesIO()
    with wave.open(wav_stream, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        for frame in frames:
            wf.writeframes(frame)
    wav_stream.seek(0)
    return wav_stream

  
# start recording_overlap handles reading the data from audio stream and feeding it to the transcription pipeline
# TODO Change chunking behaviour to be based off of sentence content.  This requires getting transcriptions back from the model server
def process_stream(stream, sample_width, input_queue, buffer_prune_queue):
    """
    Process the audio stream in chunks and perform necessary operations on each chunk.

    Args:
        stream (audio stream): The audio stream to process.
        sample_width (int): The sample width of the audio stream.
        input_queue (queue): The queue to put processed audio chunks into.
        buffer_prune_queue (queue): The queue to receive prune signals for audio buffer.

    Returns:
        None
    """  
    
    #frames holds the current audio chunk
    current_frames = []
    silence = False

    try:
        reference_time = 0
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            current_frames.append(data)
            # Calculate elapsed time
            elapsed_time = len(current_frames) * CHUNK / RATE

            # TODO Implement silence detection
            if (elapsed_time - reference_time >= AUDIO_CHUNK_LENGTH) or silence == True:
                
                #buffer_prune queue will get pushed to it the time where the audio buffer should be pruned when its appropriate
                #TODO: Implement sentinel value for checking queue emptyness instead of .empty()?
                if not buffer_prune_queue.empty():
                    #TODO: profile how long it takes to prune the buffer with list slicing
                    prune_signal = buffer_prune_queue.get()
                    current_frames = current_frames[int(prune_signal * RATE / CHUNK):]
                
                reference_time = elapsed_time
                # Prepare the chunk for processing
                processing_frames = current_frames

                # Process the chunk
                wav_stream = process_audio_chunk(processing_frames, sample_width)
                input_queue.put(wav_stream)
            
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        return
        
#- End Audio processing functions -#

def main():
    
    ###---------Setup Audio Stream---------###
    pa = pyaudio.PyAudio()
    sample_width = pa.get_sample_size(FORMAT)
    stream = pa.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=CHUNK)
    ###---------End Setup Audio Stream---------###

    ###--------- Multiprocessing Setup ---------###
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    buffer_prune_queue = multiprocessing.Queue()

    server_process = multiprocessing.Process(target=model_server, args=(input_queue, output_queue, buffer_prune_queue))
    display_process = multiprocessing.Process(target=output_transcript, args=(output_queue,))

    server_process.start()
    display_process.start()
    
    ###--------- End Multiprocessing Setup ---------###

    #Wait for user input to start recording
    input("Program started. Press any key to start recording.  Press Ctrl+C to stop.")
    output_queue.put((str(0),"Beginning transcription! \n",""))
    process_stream(stream, sample_width, input_queue, buffer_prune_queue)

    #send custom chunk id
    input_queue.put(None)  # Signal model process to shut down
    output_queue.put(None)  # Signal display process to shut down


    server_process.join()
    display_process.join()
    #Cleanup 
    server_process.terminate()
    display_process.terminate()

    stream.stop_stream()
    stream.close()
    pa.terminate()

######------ End Functions ------###### 

if __name__ == "__main__":
    main()


    