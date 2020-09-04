from google.cloud import storage
import os
import json
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import types
import time


#Replace with your Google credentials json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='speech-eye-gaze.json'

#Replace with your Google storage bucket name
bucket_name = "dictation_eye_gaze_bucket_name"


def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def transcribe_gcs_with_word_time_offsets(gcs_uri):
    'Taken from https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/speech/cloud-client/transcribe_word_time_offsets.py'
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""


    from google.protobuf.json_format import MessageToDict

    client = speech.SpeechClient()

    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        audio_channel_count=2,
        # encoding='LINEAR16',
        language_code='en-US',
        use_enhanced=True,
        enable_word_time_offsets=True,
        model='default',
        enable_automatic_punctuation=True,
        speech_contexts = [{'phrases': ['atelectasis','aorta','patchy','opacity','cardio','effusion','Basilar','comma','cardiomegaly','pic','superior vena cava','hijau','hijau line','junction','pericardial','vena cava','costrophrenic','quartant', 'is', 'prominent','mild', 'lungs', 'normal', 'lungs are normal', 'aortic', 'heart']}]  # Note the change in the field
    #You can add more phrases
    )

    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    # result = operation.result(timeout=180)
    time.sleep(2)
    result = operation.result()


    transcript = ""
    sentences = dict()
    counter = 1
    for result in result.results:
        sentence = result.alternatives[0]
        print(u'Transcript: {} '.format(sentence.transcript),'Confidence: {}'.format(sentence.confidence))
        transcript = transcript+' '+sentence.transcript
        words = []
        for word_info in sentence.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time


            print('Word: {}, start_time: {}, end_time: {}'.format(
                word,
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9))

            word_dict = {'word':word, 'begin_time':start_time.seconds + start_time.nanos * 1e-9,'end_time':end_time.seconds + end_time.nanos * 1e-9}
            words.append(word_dict)
        sentence_name = 'sentence_'+str(counter)
        sentences[sentence_name]=words
        counter +=1

    sentences["full_transcript"]=transcript


    return transcript, sentences


if __name__ == '__main__':
    #Replace with the location of the audio_segmentation_transcripts. See readme file.
    folder = 'audio_segmentation_transcripts'

    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    for subfolder in subfolders:
        try:
            audio_file = os.path.join(subfolder,'audio.wav')
            upload_blob(audio_file, "case.wav")
            transcript, sentences = transcribe_gcs_with_word_time_offsets("gs://"+bucket_name+"/case.wav")
            with open(os.path.join(subfolder,'original_transcript.json'), "w") as jsonFile:
                json.dump(sentences, jsonFile, indent=4, sort_keys=True)
            jsonFile.close()
            print('Finished speech-to-text for ',subfolder)
        except:
            print("Error with case: ", subfolder)
