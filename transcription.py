from flask import Flask, request, jsonify
import whisper
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])  # Permite CORS para qualquer origem

# Carregar o modelo Whisper (use "small" ou "medium" dependendo do desempenho desejado)
modelo = whisper.load_model("medium")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():

    try:
        # Verifica se o arquivo de áudio foi enviado
        if 'audio' not in request.files:
            return jsonify({"error": "Arquivo de áudio não encontrado."}), 400

        # Recebe o arquivo de áudio enviado
        audio_file = request.files['audio']

        # Lê o conteúdo do arquivo de áudio
        audio_data = audio_file.read()

        # Salva o arquivo temporariamente para processar com Whisper
        temp_file_path = "temp_audio.mp3"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_data)

        # Transcrever o áudio com Whisper
        result = modelo.transcribe(temp_file_path)

        # Remover o arquivo temporário
        os.remove(temp_file_path)

        # Retornar a transcrição
        
        return jsonify({"transcription": result['text']})

    except Exception as e:
        # Se ocorrer algum erro, retornar o erro detalhado
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)
