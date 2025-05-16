from flask import Flask, request, jsonify
from sklearn.multiclass import OneVsRestClassifier
from flask_cors import CORS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])# Permite CORS para todas as rotas
# CORS(app)  # Permite CORS para todas as rotas

# Dados de entrada (relatos de violência)
data = [
    # Violência Física
    "Ela foi agredida fisicamente pelo marido durante uma discussão.",
    "Ele me bateu durante uma discussão. Eu estava com medo e não sabia o que fazer.",
    "Ela me deu um tapa no rosto e me empurrou para fora de casa.",
    "Ele me agrediu com socos e me dizia que eu merecia isso.",
    "Ele me empurrou contra a parede e quase me estrangulou durante a briga.",
    "Ele me agrediu com um objeto, deixando marcas no meu corpo.",
    "Ela me empurrou tão forte que caí no chão, machucando minhas costas.",
    "Sempre que tentava sair de casa, ele me impedia fisicamente.",
    "Ele me atacou enquanto dormia, me acordando com um soco no rosto.",
    "Ele me bateu com um cinto enquanto discutíamos.",
    "Ela me agrediu enquanto estava tentando fugir da situação.",
    "Ele me bateu com um pedaço de madeira.",
    "Ele sempre me agride fisicamente quando perde a calma.",
    "Durante uma discussão, ele me empurrou para fora de casa.",
    "Ele quebrou uma cadeira em cima de mim durante uma briga.",
    "Ele me agrediu enquanto eu estava tentando sair de casa.",
    "Ela me jogou contra a parede durante uma discussão.",
    "Ele me ameaçou com uma faca durante uma briga.",
    "Eu fui socada por ele em frente aos meus filhos.",
    "Ele me arrastou pelo cabelo durante uma briga.",
    "Ele me bateu tanto que eu não consegui ir trabalhar.",
    "Ela me deu um tapa no rosto durante um jogo de palavras.",
    "Ele me forçou a fazer tarefas domésticas sob ameaça de violência física.",
    "Ele me agrediu porque não concordei com sua opinião.",
    "Ele usou uma barra de ferro para me agredir.",
    "Durante a noite, ele me acordou com um soco.",
    "Ele me agrediu durante uma viagem em que não podíamos sair do local.",
    "Ele me bateu depois de uma simples discussão sobre dinheiro.",
    "Ele me jogou contra o ventilador",
    "Ele me empurrou para o chão e me deixou lá.",
    "Ele queimou meu braço.",
    "Ele raspou minha cabeça.",

    # Violência Psicológica
    "Ele sempre me chama de burra e diz que eu nunca seria nada sem ele.",
    "Ela me humilha publicamente, fazendo com que eu me sinta inútil.",
    "Eu fico me perguntando se algum dia vou ser boa o suficiente para ele.",
    "O abuso psicológico é pior do que qualquer agressão física. Ele me faz sentir invisível.",
    "O controle emocional que ele exerce sobre mim é devastador. Eu vivo com medo.",
    "Ele me faz sentir culpada por tudo, mesmo quando não tenho culpa.",
    "Ela me diz que ninguém mais vai me amar, que sou feia e sem valor.",
    "Ele sempre tenta me fazer acreditar que sou incapaz de fazer as coisas.",
    "O abuso emocional é uma constante. Ele diz que ninguém mais me suportaria.",
    "Ela constantemente me chama de fraca e diz que nunca encontrarei outra pessoa.",
    "Ele sempre me chama de burra e faz piada com minhas opiniões.",
    "Ela me pressiona psicologicamente, dizendo que jamais vou ser boa o suficiente.",
    "Ele me diz que estou velha demais para mudar minha vida.",
    "Ela me diz que sou inútil e que só ele pode me dar valor.",
    "Ele diz que ninguém mais me amará e que eu estou fadada à solidão.",
    "Ele me faz sentir que sou uma pessoa incapaz e sem valor.",
    "Ela me chama de burra e incompetente para qualquer tarefa.",
    "Ele constantemente me culpa por coisas que não tenho controle.",
    "Ela faz questão de me lembrar de todos os meus erros do passado.",
    "Ele me manipula emocionalmente para que eu sempre faça o que ele quer.",
    "Ele constantemente me humilha na frente de outras pessoas.",
    "Ela me faz sentir que não sou capaz de realizar nada sem a ajuda dela.",
    "Ele me diz que sou fraca e que ele é a única pessoa que pode me ajudar.",
    "Ela me faz sentir que o mundo está contra mim e que ninguém me entende.",
    "Ele me diz que eu nunca vou encontrar um homem melhor do que ele.",
    "Ele me ameaça emocionalmente dizendo que se eu sair, vai ser pior para mim.",
    "Ela diz que não tenho força para seguir em frente, que vou fracassar.",
    "Ele me humilha diante de familiares e amigos.",
    "Ela me faz sentir como se não fosse capaz de tomar decisões por mim mesma.",
    "Ele me disse que ninguém nunca irá me querer",
    "Ele me chama de feia e diz que ninguém gosta de mim",
    "Ele diz que eu nunca vou encontrar alguém melhor do que ele"

    # Ameaça
    "Ele me ameaçou de morte se eu tentasse deixar ele.",
    "Ela disse que me destruiria se eu denunciasse as agressões.",
    "Ele me prometeu que, se eu fosse embora, ele faria minha vida um inferno.",
    "Ele sempre me diz que, se eu contar para alguém, vou me arrepender.",
    "Me sinto impotente. Ele me ameaça todos os dias com palavras cruéis.",
    "Ele jurou que, se eu denunciasse, ele me mataria e à minha família.",
    "Ela disse que nunca mais veria meus filhos se eu o deixasse.",
    "Ele me disse que eu seria encontrada morta se falasse com a polícia.",
    "Ele me ameaça constantemente, dizendo que vai me perseguir até o fim.",
    "Ele me disse que, se eu denunciasse, minha vida acabaria.",
    "Ele me ameaçou dizendo que iria destruir minha carreira e minha vida social.",
    "Ela disse que faria com que todos acreditassem que eu estava mentindo.",
    "Ele me prometeu que, se eu o deixasse, ele me faria sofrer.",
    "Ele jurou que nunca mais veria meus filhos se eu falasse sobre o que acontece.",
    "Ele disse que iria contar mentiras sobre mim, destruindo minha reputação.",
    "Ele me ameaçou de morte dizendo que faria isso de forma silenciosa.",
    "Ele me disse que ninguém acreditaria em mim se eu denunciasse suas ameaças.",
    "Ele prometeu que me faria perder tudo o que conquistei até hoje.",
    "Ele me ameaçou dizendo que iria me machucar se eu tentasse fugir.",
    "Ela disse que iria destruir minha vida, começando por minha família.",
    "Ele sempre diz que, se eu contar para alguém, vou me arrepender.",
    "Ele me ameaça com vingança caso eu tente denunciá-lo.",
    "Ele me disse que faria o inferno na minha vida se eu tentasse sair.",
    "Ela me disse que iria espalhar mentiras sobre mim caso eu o deixasse.",
    "Ele me disse que faria minha vida miserável caso eu contasse para alguém.",
    "Ele me ameaçou com palavras de morte e punições severas caso eu fugisse.",
    "Ele jurou que tomaria tudo de mim se eu o deixasse.",
    "Ele disse que se eu saísse de casa hoje eu iria ver",
    "Ele disse que eu ia ver o que era bom pra tosse",
    "Ele disse que mataria meus filhos se eu denunciasse"

    # Violência Sexual
    "Fui estuprada por meu ex-marido e ele sempre me culpava por isso.",
    "Ele me forçou a ter relações sexuais sem meu consentimento e depois me disse que ninguém acreditaria.",
    "Durante uma briga, ele me agrediu fisicamente e me forçou a fazer sexo com ele.",
    "Ele me violou em casa, dizendo que eu era dele e que não poderia ir embora.",
    "Fui estuprada durante um encontro forçado. Ele não quis ouvir meus pedidos para parar.",
    "Ele me obrigou a fazer sexo com ele, apesar de eu ter pedido para parar.",
    "Durante um jogo de manipulação, ele me forçou a fazer algo que eu nunca teria feito.",
    "Ele me agrediu e depois forçou uma relação sexual, dizendo que era normal.",
    "Ele usou a força para me obrigar a manter relações sexuais, ignorando meus gritos.",
    "Ele me forçou a fazer sexo após uma discussão, e depois se fez de inocente.",
    "Ele me forçou a ter relações sexuais mesmo quando eu estava chorando e pedindo para parar.",
    "Ele me estuprou e depois disse que eu não tinha direito de recusar.",
    "Ele sempre dizia que, como meu marido, ele tinha direito a sexo a qualquer hora.",
    "Ele me forçou a ter relações sexuais em frente aos meus filhos.",
    "Ele me disse que ninguém acreditaria se eu falasse sobre o abuso.",
    "Ele me ameaçou de morte se eu falasse sobre o abuso sexual.",
    "Ele se aproveitou de mim quando eu estava inconsciente, dizendo que eu pedi por isso.",
    "Ele me forçou a ter relações sexuais enquanto me fazia sentir culpa por não gostar.",
    "Ele me estuprou em várias ocasiões, dizendo que ninguém mais me amaria.",
    "Ele insistiu em fazer sexo comigo, mesmo quando eu estava claramente desconfortável.",
    "Ele me forçou a fazer sexo quando eu estava doente, dizendo que não havia desculpas.",
    "Ele disse que, como meu marido, ele tinha o direito de fazer o que quisesse comigo.",
    "Ele me agrediu sexualmente e depois me culpou por não ser suficientemente atraente.",
    "Ele me fez fazer coisas que eu nunca teria feito sob outras circunstâncias.",
    "Ele me forçou a ter sexo com ele, ignorando totalmente minha vontade.",
    "Ele me estuprou em diversas ocasiões e dizia que ninguém acreditaria em mim.",
    "Ele abusou de mim durante uma festa e me forçou a ficar em silêncio sobre o ocorrido.",
    "Ele continuou mesmo quando eu pedi para ele parar"
]

# Inicializando a lista de rótulos de forma automatizada
labels = []

# Adicionando 30 rótulos para cada tipo de violência
labels.extend([0] * 30)  # Violência Física
labels.extend([1] * 30)  # Violência Psicológica
labels.extend([2] * 30)  # Ameaça
labels.extend([3] * 30)  # Violência Sexual

# Criar DataFrame
df = pd.DataFrame({'text': data, 'label': labels})

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Carregar o modelo BERTimbau e o tokenizer
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

# Função para transformar textos em embeddings com BERT (usando o token [CLS])
def encode_texts(texts):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
    # Pegando a representação do token [CLS] como vetor fixo
    return outputs.last_hidden_state[:, 0, :].numpy()  # last_hidden_state é a representação da camada final

# Transformar os textos em embeddings
X_train_encoded = encode_texts(X_train)
X_test_encoded = encode_texts(X_test)

# Criar pipeline com o modelo BERT e o classificador SVM
clf = OneVsRestClassifier(SVC(kernel='linear'))
# Treinar o modelo
clf.fit(X_train_encoded, y_train)

# Função de predição
def predict(text):
    # Transformar o novo dado em embedding
    text_encoded = encode_texts(pd.Series([text]))
    prediction = clf.predict(text_encoded)
    return prediction[0]

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    text = data['text']
    
    # Predição
    prediction = predict(text)
    
    # Retornar a classificação
    labels = ["Violência Física", "Violência Psicológica", "Ameaça", "Violência Sexual"]
    return jsonify({"result": labels[prediction]})

if __name__ == '__main__':
    app.run(debug=True)

# Avaliar o modelo
# y_pred = clf.predict(X_test_encoded)
# print(classification_report(y_test, y_pred, target_names=["Violência Física", "Violência Psicológica", "Ameaça", "Violência Sexual"]))