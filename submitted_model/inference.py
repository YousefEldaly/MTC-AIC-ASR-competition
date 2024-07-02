import os
import pandas as pd
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel

def install_dependencies():
    os.system('pip install wget')
    os.system('apt-get install sox libsndfile1 ffmpeg -y')
    os.system('pip install text-unidecode')
    os.system('pip install matplotlib>=3.3.2')
    os.system('pip install git+https://github.com/NVIDIA/NeMo.git@main')
    os.system('pip install nemo_toolkit[asr]')
    os.system('pip install pandas')

def transcribe_files(checkpoint_path, test_dir, submission_file):
    # Restore the model from the checkpoint
    model = EncDecCTCModel.restore_from(restore_path=checkpoint_path)
    
    # Assuming you have a list of WAV files in your test directory
    wav_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.wav')]

    transcriptions = []
    for wav_file in wav_files:
        audio_id = os.path.splitext(os.path.basename(wav_file))[0]  # Get audio file ID without extension
        transcription = model.transcribe(audio=wav_file)  # Replace with your actual transcription function
        transcriptions.append({'audio': audio_id, 'transcript': transcription})

    # Create a DataFrame from transcriptions
    df = pd.DataFrame(transcriptions)

    # Save DataFrame to CSV
    df.to_csv(submission_file, index=False)

if __name__ == "__main__":
    # Install dependencies
    install_dependencies()

    # Define paths
    checkpoint_path = 'model.nemo'  # Adjust this path as needed
    test_dir = './test/'  # Adjust this path as needed
    submission_file = 'submissionFromThirtyyy.csv'  # Adjust this path as needed

    # Run transcription
    transcribe_files(checkpoint_path, test_dir, submission_file)
