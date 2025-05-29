from flask import Flask, request, jsonify
from sklearn.multiclass import OneVsRestClassifier
from flask_cors import CORS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])# Permite CORS para todas as rotas
# CORS(app)  # Permite CORS para todas as rotas

class_names = [
    "Ameaça por violência doméstica ou familiar",      # Rótulo 1
    "Calúnia por violência doméstica ou familiar",     # Rótulo 2
    "Difamação por violência doméstica ou familiar",   # Rótulo 3
    "Injúria por violência doméstica ou familiar",     # Rótulo 4
    "Constrangimento ilegal por violência doméstica ou familiar", # Rótulo 5
    "Cárcere privado por violência doméstica ou familiar",     # Rótulo 6
    "Descumprimento de medida protetiva de urgência",  # Rótulo 7
    "Perturbação do sossego por violência doméstica ou familiar", # Rótulo 8
    "Lesão corporal",                                  # Rótulo 9
    "Tentativa de feminicídio",                        # Rótulo 10
    "Estupro",                                         # Rótulo 11
    "Sequestro"                                        # Rótulo 12
]

data = [
    # Rótulo 1 - Ameaça por violência doméstica ou familiar
    "Ele me disse bem claro: 'Se você abrir a boca pra contar pra alguém o que acontece aqui, você vai se arrepender amargamente. Ninguém vai te achar'.",
    "Recebi uma mensagem no celular que dizia: 'É melhor você ficar bem quietinha e não me provocar, senão o próximo a te visitar vai ser o capeta. E eu vou fazer questão de te mandar pra ele'.",
    "Durante a briga, ele pegou uma faca na cozinha e berrou que da próxima vez que eu 'saísse da linha', ele não ia pensar duas vezes antes de me cortar inteira.",
    "Minha ex não aceita o término. Vive mandando áudio falando que, se eu arrumar outra pessoa, ela vai fazer da minha vida e da vida dessa pessoa um inferno, que ela sabe onde eu moro e trabalho.",
    "Ouvi ele falando no telefone com alguém, rindo: 'Pode deixar, essa aí vai ter o que merece. Se ela acha que vai ficar com a casa, tá muito enganada. Vai aprender na marra quem manda'.",
    "Depois que pedi a separação, ele começou a me ligar de madrugada, só pra dizer que eu ia pagar caro pela 'afronta' e que era pra eu ter medo até da minha própria sombra.",
    "Ele escreveu num bilhete e deixou na minha bolsa: 'Tô de olho em cada passo seu. Um deslize e você já era. Pensa bem nos seus filhos antes de fazer qualquer besteira'.",
    "Falou que se eu não voltasse pra ele, ia colocar fogo no meu carro com minhas coisas dentro. Disse que eu não perdia por esperar.",
    "Mandou foto de uma arma pelo WhatsApp com a legenda 'saudade de você'. Fiquei apavorada, entendi como um aviso.",
    "Ela me mandou uma caixa com um rato morto e um bilhete: 'O próximo pode ser você'.",
    "Ele disse que se eu não retirasse a queixa, ele faria da vida dos meus pais um inferno.",
    "Recebi uma coroa de flores em casa com uma fita escrito 'Descanse em paz, [meu nome]'.",
    "Ele me mostrou conversas que provavam que ele sabia onde meus filhos estudavam e os horários deles, e disse 'seria uma pena se algo acontecesse'.",
    "Ela vive postando indiretas nas redes sociais com fotos de armas e frases como 'a vingança é um prato que se come frio'.",
    "Ele me ligou usando um modificador de voz, dizendo que meus dias estavam contados.",
    "Deixou um boneco de vodu na minha porta com agulhas espetadas, claramente uma ameaça macabra.",
    "Ele me disse que se eu não voltasse pra casa até o fim do dia, ele ia 'dar um jeito' no meu gato de estimação.",
    "Ela enviou um e-mail para meu chefe com informações falsas e ameaçou enviar mais se eu não fizesse o que ela queria.",
    "Ele me cercou com o carro e fez gestos como se fosse atirar, rindo da minha cara de pavor.",
    "Mandou um motoboy entregar um envelope na minha casa só com uma bala de revólver dentro.",
    "Ele disse que conhecia gente perigosa e que se eu 'pisasse na bola', ele ia 'encomendar um serviço'.",
    "Ela me ameaçou de expor fotos íntimas minhas para toda a lista de contatos do meu celular se eu bloqueasse o número dela.",
    "Ele me mostrou um canivete e disse: 'Isso aqui tem o seu nome. É melhor você andar na linha'.",
    "Recebi uma mensagem de um perfil falso com uma foto da fachada da minha casa e a legenda 'Bela vista, não?'.",
    "Ele disse que se eu o denunciasse, ele faria com que eu perdesse a guarda dos meus filhos, usando 'métodos sujos'.",
    "Ela me ameaçou de jogar ácido no meu rosto se me visse com outra pessoa.",
    "Ele me disse que ia me 'caçar como um animal' se eu tentasse me esconder dele.",
    "Deixou um bilhete no para-brisa do meu carro: 'Seu tempo está acabando. Aproveite enquanto pode'.",
    "Ele me ligou e descreveu exatamente a roupa que eu estava usando, mesmo eu estando dentro de casa, para mostrar que estava me vigiando.",
    "Ela ameaçou quebrar todos os meus instrumentos de trabalho, me impedindo de ganhar meu sustento.",
    "Ele disse que ia 'fazer uma visita' para minha mãe idosa se eu não parasse de 'falar mal' dele.",
    "Recebi um vídeo dele afiando uma faca e olhando para a câmera com um sorriso sinistro.",
    "Ela me disse: 'Você não perde por esperar a surpresinha que preparei para o seu aniversário'.",
    "Ele me ameaçou, dizendo que se algo acontecesse com ele na prisão (por causa de outra queixa minha), os 'amigos' dele cuidariam de mim aqui fora.",
    "Ela postou uma foto minha dormindo, que só poderia ter sido tirada de dentro da minha casa, com a legenda 'Sonhos tranquilos?'.",
    "Ele disse que ia me 'fazer engolir cada palavra' que eu disse sobre ele, e que não seria de forma gentil.",
    "Mandou um recado por um amigo em comum: 'Diz pra ela que a paciência do palhaço acabou'.",
    "Ela me ameaçou dizendo que, se eu não desbloqueasse ela nas redes sociais, ela iria até a minha faculdade fazer um escândalo.",
    "Ele me disse: 'Reza pra gente não se cruzar na rua, porque se isso acontecer, só um de nós vai sair andando'.",

    # Rótulo 2 - Calúnia por violência doméstica ou familiar
    "Ele foi na delegacia e registrou um boletim de ocorrência falso, me acusando de ter roubado as joias da mãe dele, o que nunca aconteceu.",
    "Ela contou para todos os nossos amigos que eu estava vendendo drogas dentro de casa, uma mentira absurda que pode me levar pra cadeia.",
    "Meu ex-companheiro disse ao conselho tutelar que eu agrido fisicamente nosso filho, imputando-me o crime de maus-tratos, sendo que sou uma mãe zelosa.",
    "Recebi uma intimação porque ela me acusou falsamente de ter cometido estelionato, usando meu nome para fazer compras fraudulentas. Tive que provar minha inocência.",
    "Ele espalhou para os vizinhos que eu sou receptador de mercadoria roubada, e que a polícia já estava me investigando. Tudo invenção dele.",
    "Ela me acusou perante o síndico de ter danificado o portão do condomínio de propósito, o que configura crime de dano, mas foi um acidente causado por outra pessoa.",
    "Para não pagar a pensão, ele inventou para o juiz que eu pratico alienação parental de forma criminosa, me denunciando falsamente.",
    "Descobri que ele me acusou formalmente de ter furtado documentos da empresa dele, uma calúnia grave para me prejudicar profissionalmente.",
    "Ela me acusou falsamente de ter praticado um aborto ilegal, contando isso para pessoas da nossa igreja.",
    "Ele registrou uma ocorrência dizendo que eu o ameacei com uma arma, o que é uma invenção para conseguir uma medida protetiva contra mim.",
    "Fui intimado porque ela alegou que eu estava perseguindo-a (stalking), sendo que era ela quem me procurava.",
    "Ele espalhou que eu estava envolvido em corrupção na empresa onde trabalho, uma acusação criminal gravíssima.",
    "Ela disse ao nosso círculo de amigos que eu cometi o crime de bigamia, pois teria me casado com ela ainda estando legalmente casado com outra.",
    "Fui acusado por ele de ter cometido fraude fiscal, inventando que eu sonego impostos de forma sistemática.",
    "Ela me denunciou por um suposto sequestro do nosso filho, quando na verdade tínhamos um acordo verbal sobre a visita.",
    "Ele me atribuiu falsamente o crime de ter disseminado pornografia infantil, uma acusação nojenta e mentirosa.",
    "Ela me acusou de ter cometido o crime de tortura contra um animal de estimação que tínhamos, apenas por vingança.",
    "Ele me imputou o crime de ter incendiado o carro dele, quando na verdade o incêndio foi acidental e causado por uma pane elétrica.",
    "Fui falsamente acusado por ela de ter praticado o crime de falsidade ideológica ao preencher um documento público.",
    "Ela disse que eu cometi o crime de descaminho, trazendo produtos ilegais do exterior, uma completa invenção.",
    "Ele me acusou de ter cometido o crime de lesão corporal grave contra um parente dele, fato que nunca ocorreu.",
    "Ela me imputou o crime de ter praticado extorsão mediante sequestro contra um sócio dela, uma fantasia absurda.",
    "Ele me acusou formalmente de ter cometido o crime de peculato, desviando recursos de uma associação da qual fazíamos parte.",
    "Ela me denunciou por um suposto crime de lavagem de dinheiro, alegando que meus rendimentos eram de origem ilícita.",
    "Fui acusado por ele de ter cometido o crime de violação de direito autoral, por supostamente plagiar uma obra dele.",
    "Ela me imputou falsamente o crime de ter praticado cárcere privado contra ela mesma, distorcendo uma discussão que tivemos.",
    "Ele me acusou de ter cometido o crime de porte ilegal de arma de fogo, plantando uma arma no meu carro.",
    "Ela me denunciou por um suposto crime de tráfico de influência, alegando que usei minha posição para obter vantagens indevidas.",
    "Fui acusado por ele de ter cometido o crime de associação criminosa, dizendo que faço parte de uma quadrilha.",
    "Ela me imputou o crime de ter praticado um homicídio culposo no trânsito, sendo que eu nem estava dirigindo no dia do fato.",
    "Ele me acusou de ter cometido o crime de denunciação caluniosa contra ele, invertendo a situação.",
    "Ela me denunciou por um suposto crime de abandono de incapaz, referente aos nossos filhos, o que é uma mentira cruel.",
    "Fui acusado por ele de ter cometido o crime de estelionato sentimental, alegando que me aproximei dele apenas por interesse financeiro.",
    "Ela me imputou o crime de ter praticado um ato de terrorismo, uma acusação completamente desproporcional e falsa.",
    "Ele me acusou de ter cometido o crime de corrupção de menores, por supostamente influenciar negativamente o enteado dele.",
    "Ela me denunciou por um suposto crime de falsificação de documento particular, referente a um contrato de aluguel.",
    "Fui acusado por ele de ter cometido o crime de charlatanismo, por supostamente prometer curas milagrosas.",
    "Ela me imputou o crime de ter praticado um furto qualificado na casa de um amigo em comum, durante uma festa.",

    # Rótulo 3 - Difamação por violência doméstica ou familiar
    "Fiquei sabendo pela minha vizinha que meu ex-marido anda espalhando pelo bairro que eu o traía com vários homens e que por isso ele me 'colocou pra correr'.",
    "Descobri que ela criou um perfil falso na internet e está postando fotos minhas antigas com legendas insinuando que eu sou garota de programa e que abandonei meus filhos.",
    "Ele ligou para os meus pais e para o meu chefe contando uma história completamente distorcida sobre uma dívida antiga, me pintando como uma caloteira e uma pessoa desonesta, só pra sujar minha imagem.",
    "Minha ex-sogra está falando para todos os parentes e amigos em comum que eu sou uma péssima mãe, que sou negligente com as crianças e que só quero o dinheiro da pensão.",
    "Ele mandou mensagens para o grupo da família dela dizendo que eu roubei dinheiro dele e que sou uma pessoa de má índole, tentando me indispor com todos.",
    "No trabalho, uma colega veio me perguntar se era verdade que eu estava saindo com o chefe, porque meu ex tinha ligado lá e 'alertado' sobre meu 'comportamento inadequado'.",
    "Ela tem contado para as amigas que eu tenho problemas com álcool e que sou agressivo, fatos que não são verdadeiros, claramente para que as pessoas se afastem de mim.",
    "Ele anda dizendo na faculdade que eu só passei nas matérias porque saí com o professor. Que absurdo, minha reputação está sendo destruída por causa disso.",
    "Ouvi ela comentando com a síndica que eu dou festas barulhentas todo fim de semana e que sou uma péssima moradora, o que é uma mentira deslavada.",
    "Ele disse no grupo de WhatsApp da família que eu sou uma pessoa fútil e que só penso em aparências.",
    "Ela contou para as amigas que eu sou um péssimo amante e que não satisfaço ninguém.",
    "Fiquei sabendo que ele anda falando mal do meu trabalho, dizendo que sou desqualificado e que consegui a vaga por sorte.",
    "Ela espalhou o boato de que eu estava com uma doença contagiosa grave, para que as pessoas se afastassem de mim.",
    "Ele disse para os nossos filhos que eu não os amo de verdade e que prefiro minha nova vida de solteira.",
    "Ela comentou com a vizinhança que eu não pago minhas contas em dia e que vivo endividada.",
    "Ele está falando para os amigos dele que eu sou uma pessoa mesquinha e que não ajudo ninguém.",
    "Ela postou no Facebook que eu sou uma pessoa falsa e manipuladora, e que todos deveriam tomar cuidado comigo.",
    "Ele disse para o meu novo namorado que eu sou uma pessoa possessiva e ciumenta ao extremo.",
    "Ela contou no salão de beleza que eu fiz várias cirurgias plásticas e que nada em mim é natural.",
    "Fiquei sabendo que ele está dizendo que eu sou uma pessoa que não tem palavra e que não cumpro o que prometo.",
    "Ela espalhou que eu sou uma pessoa preguiçosa e que não gosto de trabalhar.",
    "Ele disse para a minha família que eu o abandonei em um momento difícil, o que não é verdade.",
    "Ela está contando para todos que eu sou uma pessoa arrogante e que me acho superior aos outros.",
    "Ele disse para os pais dele que eu era uma má influência para os netos.",
    "Ela fez questão de contar para o meu chefe que eu estava procurando outro emprego, para me prejudicar.",
    "Ele está espalhando que eu sou uma pessoa barraqueira e que arrumo confusão por onde passo.",
    "Ela disse para as amigas em comum que eu me visto de forma vulgar e inadequada.",
    "Ele contou para os filhos que eu traí a confiança dele diversas vezes durante o casamento.",
    "Ela está falando para a vizinhança que eu sou uma pessoa suja e que minha casa é um lixo.",
    "Ele disse para o meu senhorio que eu costumo atrasar o aluguel e que sou um inquilino problemático.",
    "Ela espalhou no meu antigo círculo de amizades que eu me afastei porque 'virei as costas' para todos.",
    "Ele está difamando minha capacidade intelectual, dizendo que sou burra e incapaz de tomar decisões.",
    "Ela contou para os colegas de faculdade que eu plagiei um trabalho importante.",
    "Ele está dizendo que eu sou uma pessoa interesseira e que só me relaciono com os outros por conveniência.",
    "Ela espalhou que eu sou uma pessoa de hábitos noturnos duvidosos e que chego tarde em casa por motivos escusos.",
    "Ele disse para a minha família que eu sou uma pessoa fria e que não demonstro afeto por ninguém.",
    "Ela está contando para todos que eu sou uma pessoa invejosa e que cobiço as coisas dos outros.",
    "Ele está difamando minha fé, dizendo que sou uma pessoa hipócrita e que não pratico o que prego.",
    "Ela espalhou no condomínio que eu sou uma pessoa desregrada e que não respeito as normas de convivência.",


    # Rótulo 4 - Injúria por violência doméstica ou familiar
    "No meio da discussão, ele cuspiu no meu rosto e gritou que eu era uma 'vagabunda imprestável' e que não servia nem pra ser lixo.",
    "Ela vive me humilhando na frente dos nossos filhos, me chamando de 'burro', 'incompetente' e dizendo que sou um 'fracassado que não deu em nada na vida'.",
    "Recebi uma mensagem dele cheia de xingamentos, me chamando de 'piranha', 'louca descontrolada' e dizendo que eu tenho 'o demônio no corpo'.",
    "Quando reclamei que ele não ajudava em casa, ele respondeu aos berros que eu era uma 'gorda nojenta' e que ele sentia repulsa só de olhar pra minha cara.",
    "Ele constantemente me compara com outras mulheres, dizendo que sou 'relaxada', 'feia' e que 'qualquer uma na rua é melhor' do que eu.",
    "Ela me olhou de cima a baixo com desprezo e disse: 'Você não passa de uma encostada, uma inútil que só serve pra gastar meu dinheiro. Tenho vergonha de você'.",
    "Toda vez que tento expressar minha opinião, ele me corta e diz: 'Cala a boca, sua anta, você não entende nada de nada mesmo'.",
    "Me mandou um áudio me chamando de 'idiota', 'sem cérebro' e 'um peso morto' só porque eu esqueci de pagar uma conta.",
    "Na frente da família dele, ela fez questão de dizer que eu sou uma 'vergonha' e que 'não sei me portar como uma mulher de verdade'.",
    "Ele me chamou de 'biscate' na frente dos meus filhos quando me recusei a dar dinheiro pra ele.",
    "Ela me disse que eu sou um 'zero à esquerda' e que minha opinião não vale nada.",
    "Recebi um e-mail dele me tratando por 'meretriz asquerosa' e outros insultos do mesmo nível.",
    "Ele me ofendeu gravemente, dizendo que eu era 'podre por dentro e por fora'.",
    "Ela me chamou de 'mão de vaca' e 'avarenta' porque não quis emprestar dinheiro a ela.",
    "Ele me disse que eu era uma 'verme' e que deveria rastejar aos pés dele.",
    "Ela me ofendeu usando termos racistas, depreciando minha cor e minha origem.",
    "Ele me chamou de 'retardado' e 'anormal' por causa de uma dificuldade de aprendizado que tenho.",
    "Ela me disse que eu era uma 'fraca' e uma 'covarde' por não reagir às provocações dela.",
    "Ele me ofendeu dizendo que eu era 'repugnante' e que sentia ânsia só de estar perto de mim.",
    "Ela me chamou de 'bruxa' e 'macumbeira' por causa das minhas crenças espirituais.",
    "Ele me disse que eu era 'feio como o diabo' e que ninguém nunca ia me querer.",
    "Ela me ofendeu chamando de 'cachorra no cio' por causa da roupa que eu estava usando.",
    "Ele me disse que eu era 'patético' e que dava pena de olhar para mim.",
    "Ela me chamou de 'traidor' e 'Judas' por ter contado um segredo dela para uma amiga.",
    "Ele me ofendeu dizendo que eu era 'um câncer' na vida dele e da família.",
    "Ela me chamou de 'ladrão' na frente de funcionários da loja, mesmo eu não tendo pego nada.",
    "Ele me disse que eu era 'um estorvo' e que ele estaria melhor se eu morresse.",
    "Ela me ofendeu me chamando de 'aleijado emocional' porque eu demonstrei tristeza.",
    "Ele me disse que eu era 'sujo' e que precisava de um banho de desinfetante.",
    "Ela me chamou de 'parasita' e 'sanguessuga' por depender financeiramente dela temporariamente.",
    "Ele me ofendeu dizendo que eu era 'um animal' e que não merecia tratamento humano.",
    "Ela me chamou de 'hipócrita' e 'falsa moralista' por criticar um comportamento dela.",
    "Ele me disse que eu era 'um erro' e que meus pais deveriam ter me abortado.",
    "Ela me ofendeu dizendo que eu era 'uma piada' e que ninguém me levava a sério.",
    "Ele me chamou de 'vadia desclassificada' após uma crise de ciúmes.",
    "Ela me disse que eu era 'uma sombra' do que já fui, tentando minar minha autoestima.",
    "Ele me ofendeu comparando minha inteligência à de uma 'porta'.",
    "Ela me chamou de 'demônio encarnado' durante uma discussão religiosa.",
    "Ele me disse que eu era 'um lixo humano' e que não merecia viver.",

    # Rótulo 5 - Constrangimento ilegal por violência doméstica ou familiar
    "Ele me obrigou a entregar meu celular e todas as minhas senhas de redes sociais, ameaçando quebrar meu notebook se eu não o fizesse imediatamente.",
    "Ela me impediu de sair de casa para ir a uma entrevista de emprego, tirou a chave da minha mão e disse que eu não precisava trabalhar, que lugar de mulher era em casa.",
    "Fui forçada por ele a assinar um documento vendendo minha parte da casa, sob ameaça de que ele sumiria com nossos filhos se eu me recusasse.",
    "Ela me constrangeu a usar roupas que eu não queria, dizendo que se eu saísse com 'aquelas roupas curtas' ela faria um escândalo na rua.",
    "Ele me proibiu de visitar minha família, ameaçando contar um segredo meu para todos eles caso eu desobedecesse.",
    "Fui coagida a apagar todas as minhas contas em redes sociais porque ele disse que não queria 'homem nenhum de olho' no que era dele.",
    "Ela me obrigou a cozinhar para os amigos dela mesmo eu estando doente, dizendo que se eu não levantasse da cama ia ser pior para mim.",
    "Ele me forçou a mentir para a polícia sobre a origem de um ferimento que ele mesmo me causou, ameaçando minha mãe caso eu contasse a verdade.",
    "Ele me forçou a gravar um vídeo pedindo desculpas por algo que não fiz, ameaçando postar fotos íntimas minhas.",
    "Ela me obrigou a usar um localizador GPS no celular para que ela soubesse onde estou o tempo todo.",
    "Fui constrangido a não ir a um casamento de um amigo porque ele não gostava da pessoa.",
    "Ele me obrigou a parar de falar com minha melhor amiga, dizendo que ela estava 'enchendo minha cabeça'.",
    "Ela me forçou a fazer uma declaração falsa em um processo judicial para protegê-la.",
    "Fui obrigado a vender um bem pessoal meu para pagar uma dívida dele, sob ameaça de agressão.",
    "Ele me impediu de usar o carro da família para ir trabalhar, dizendo que eu deveria ir a pé.",
    "Ela me constrangeu a não aceitar uma promoção no trabalho porque isso significaria viajar mais.",
    "Fui forçado a escrever uma carta de amor para ele, mesmo não sentindo mais nada, para 'provar' meu afeto.",
    "Ele me obrigou a excluir todos os contatos masculinos das minhas redes sociais.",
    "Ela me impediu de estudar para um concurso, escondendo meus livros e fazendo barulho.",
    "Fui constrangido a mentir para meus pais sobre o motivo de não poder visitá-los.",
    "Ele me obrigou a assistir filmes pornográficos com ele, mesmo eu expressando meu desconforto.",
    "Ela me forçou a participar de um esquema ilegal que ela montou, ameaçando me denunciar se eu não colaborasse.",
    "Fui obrigado a não usar determinadas cores de roupa porque ele associava a 'outras intenções'.",
    "Ele me impediu de procurar ajuda psicológica, dizendo que 'problemas de casal se resolvem em casa'.",
    "Ela me constrangeu a dar dinheiro a ela toda semana, mesmo eu não tendo condições.",
    "Fui forçado a não frequentar a igreja que eu gostava porque ele não aprovava.",
    "Ele me obrigou a fazer todas as tarefas domésticas sozinho, mesmo ele estando em casa sem fazer nada.",
    
    # Rótulo 6 - Cárcere privado por violência doméstica ou familiar
    "Ele me manteve trancada no quarto por dois dias inteiros, sem celular e sem poder sair nem para ir ao banheiro direito, só me dava água e um pouco de comida.",
    "Depois de uma briga feia, ela pegou todas as chaves da casa e do portão, me deixou presa lá dentro e disse que eu só sairia quando ela voltasse e quisesse.",
    "Fui impedida de sair do carro. Ele ficou rodando comigo pela cidade por horas, contra a minha vontade, gritando e me ameaçando para eu não terminar o relacionamento.",
    "Ele trancou todas as portas e janelas, escondeu as chaves e me proibiu de sair para encontrar minhas amigas, me mantendo em cárcere em minha própria casa.",
    "Ela me prendeu na varanda do apartamento sob sol forte por uma tarde inteira porque eu me recusei a dar dinheiro para ela.",
    "Quando tentei ir embora da casa dele, ele bloqueou a porta com o corpo e com móveis, me segurando lá dentro à força por toda a noite.",
    "Fiquei presa no banheiro por ele durante horas, como castigo por ter 'respondido mal'.",
    "Ele me levou para um sítio afastado e tirou a chave do carro, me deixando lá isolada e sem ter como voltar para a cidade por um fim de semana inteiro.",
    "Ele sabotou meu carro para que eu não pudesse sair de casa, me deixando ilhada.",
    "Ela me trancou no closet e escondeu a chave, fiquei lá por um longo tempo até ela resolver abrir.",
    "Fui mantido preso na casa de campo dele, sem acesso a telefone ou internet, durante uma semana.",
    "Ele me impediu de sair do quarto colocando um armário pesado na frente da porta.",
    "Ela me deixou trancada no carro sob o sol quente, com os vidros fechados, por quase uma hora.",
    "Fui impedido de sair para trabalhar porque ele escondeu todas as chaves do meu carro e da casa.",
    "Ele me trancou na sacada do apartamento, que não tinha como sair, durante uma noite fria.",
    "Ela me prendeu na biblioteca da casa, um cômodo sem janelas para a rua, por um dia inteiro.",
    "Fui mantido em cárcere no barco dele, ancorado em uma ilha deserta, por dois dias.",
    "Ele me trancou no sótão da casa, um lugar escuro e cheio de poeira, como forma de punição.",
    "Ela me deixou presa no banheiro de um bar, bloqueando a porta por fora, após uma discussão.",
    "Fui impedido de sair da casa dos pais dele, que moram em outra cidade, pois ele levou embora meu dinheiro e documentos.",
    "Ele me trancou em um quarto de hotel sem meus pertences e disse que só me liberaria se eu concordasse com suas condições.",
    "Ela me deixou trancada no carro dela em um estacionamento subterrâneo, sem sinal de celular.",
    "Fui mantido preso no estúdio de música dele, um local com isolamento acústico, para que ninguém ouvisse meus pedidos de ajuda.",
    "Ele me impediu de sair da cabana onde estávamos acampados, escondendo minhas botas e casaco no meio da noite fria.",
    "Ela me trancou no trailer da família durante uma viagem, me deixando sem água ou comida por horas.",
    "Fui encarcerado por ele na adega da casa, um local úmido e frio, por ter 'desobedecido' uma ordem dele.",
    "Ele me deixou preso no terraço do prédio, sem chave para voltar, sabendo que eu tinha medo de altura.",
    
    # Rótulo 7 - Descumprimento de medida protetiva de urgência
    "Mesmo com a medida protetiva que o proíbe de se aproximar a menos de 300 metros, ele apareceu na porta do meu trabalho e ficou me encarando de longe.",
    "Eu tenho uma ordem judicial que impede ele de fazer qualquer tipo de contato, mas ele continua me mandando mensagens de texto e e-mails de contas falsas.",
    "Ele foi até a escola do nosso filho no horário da saída, sendo que a medida protetiva diz claramente que ele não pode frequentar os lugares que eu e a criança costumamos ir.",
    "Apesar da restrição de não poder ligar, ele me ligou diversas vezes de um número privado, xingando e ameaçando, descumprindo a decisão do juiz.",
    "Minha ex-parceira, que está proibida de vir à minha casa, pulou o muro do quintal durante a noite e ficou batendo na minha janela.",
    "Ele enviou flores e um 'presente' para minha casa através de um amigo, o que é uma forma de contato indireto e viola a medida protetiva.",
    "Fui informada por amigos que ele estava perguntando sobre minha rotina e meus novos endereços, o que demonstra que ele não está respeitando a ordem de afastamento.",
    "Ele comentou em todas as minhas fotos novas nas redes sociais usando um perfil fake, sendo que a medida o proíbe de qualquer manifestação online direcionada a mim.",
    "Ele apareceu na igreja que frequento, mesmo a medida o proibindo de estar em locais de meu culto religioso.",
    "Recebi uma ligação do advogado dele tentando me coagir a retirar a medida, o que é uma forma de contato proibido.",
    "Ele mandou um presente de aniversário para nosso filho com um bilhete para mim, descumprindo a ordem de não comunicação.",
    "Minha ex criou um perfil falso e começou a curtir e comentar as postagens dos meus amigos mais próximos, claramente para me intimidar.",
    "Ele foi visto rondando a creche do nosso bebê, local que ele está expressamente proibido de se aproximar.",
    "Apesar da medida, ele continua a me enviar dinheiro não solicitado via PIX com mensagens provocativas na descrição.",
    "Ele tentou contato com meus colegas de trabalho para obter informações sobre meus horários e rotina.",
    "Descobri que ele pagou um detetive particular para me seguir, o que viola o espírito da medida de afastamento.",
    "Ele deixou flores no túmulo da minha mãe, um lugar que frequento e que ele sabe ser importante para mim, como forma de provocação.",
    "Minha ex publicou 'stories' no Instagram mostrando que estava em frente ao meu prédio, com legendas ameaçadoras.",
    "Ele tentou se matricular na mesma aula de ioga que eu, mesmo sabendo que a medida o impede de frequentar os mesmos ambientes.",
    "Recebi uma encomenda em casa com remetente desconhecido, mas com objetos que só ele saberia que me assustariam.",
    "Ele se aproximou do meu novo parceiro em um bar e fez ameaças veladas, uma forma de me atingir indiretamente.",
    "Apesar de estar proibido de se aproximar do meu local de estudo, ele foi visto no campus da minha universidade.",
    "Ele enviou um e-mail para minha caixa de spam com um vírus, e a mensagem continha provocações sobre a medida.",
    "Minha ex tentou registrar nosso filho em uma atividade extracurricular que ela sabia que eu frequentava com ele.",
    "Ele me enviou uma solicitação de amizade de um perfil com nome e foto falsos, mas que eu identifiquei ser ele.",
    "A medida o proíbe de portar arma, mas ele postou foto com uma arma nova nas redes sociais, me desafiando.",
    "Ele ligou para o pet shop onde levo meu cachorro e tentou marcar um horário no mesmo dia e hora que eu.",
    "Ela entrou em contato com meu terapeuta, tentando obter informações sobre meu tratamento, descumprindo a ordem judicial.",

    # Rótulo 8 - Perturbação do sossego por violência doméstica ou familiar
    "Ele fica fazendo barulho de madrugada de propósito, batendo panelas, arrastando móveis e gritando palavrões, só para me infernizar e não me deixar dormir em paz.",
    "Ela coloca o som no volume máximo, com músicas que sabe que eu detesto ou com letras ofensivas, virado para a minha janela, especialmente quando estou tentando trabalhar em casa.",
    "Toda noite, quando ele chega bêbado, começa uma sessão de gritaria e xingamentos que ecoam pela casa toda, perturbando a mim e às crianças.",
    "Meu vizinho de cima, que é meu ex, fica sistematicamente batendo no chão com força, principalmente no meu quarto, durante a madrugada, para me provocar.",
    "Ela tem o hábito de ficar acelerando a moto na frente da minha casa em horários impróprios, com o escapamento adulterado, apenas para me irritar.",
    "Ele deixa o cachorro latindo sem parar no quintal, dia e noite, e sei que é de propósito porque só acontece depois que a gente discute.",
    "Quando sabe que estou com visita, ela começa a fazer escândalo na área comum do prédio, falando alto sobre nossa vida particular para me constranger.",
    "Ele fica ligando para o meu interfone de madrugada e desligando, várias vezes seguidas, apenas para me acordar e me deixar nervosa.",
    "Ele começou a praticar bateria no apartamento dele, que é ao lado do meu, sempre nos horários que sabe que estou descansando ou trabalhando.",
    "Ela fica arrastando correntes pesadas no chão do andar de cima, produzindo um barulho metálico e arrastado que não me deixa concentrar.",
    "Ele tem o costume de fazer obras ruidosas em casa, como quebrar paredes, nos finais de semana bem cedo, perturbando todo o prédio.",
    "Ela deixa o alarme do celular dela tocando por horas de manhã cedo, mesmo quando ela já levantou, só para me acordar.",
    "Ele fica gritando o nome do time dele e xingando o juiz em frente à televisão em volume altíssimo, mesmo sabendo que as paredes são finas.",
    "Ela organiza cultos religiosos em casa com cânticos e instrumentos muito altos, que se estendem pela madrugada.",
    "Ele fica dando socos e chutes em um saco de pancadas pendurado na parede que divide nossos quartos, em horários de silêncio.",
    "Ela deixa os netos pequenos brincarem com brinquedos extremamente barulhentos no corredor, como carrinhos de rolimã e apitos.",
    "Ele tem o hábito de testar a buzina de diferentes veículos na garagem do prédio, produzindo um festival de sons irritantes.",
    "Ela fica conversando aos berros no telefone na varanda, contando detalhes íntimos da vida dela e dos outros, para que todos ouçam.",
    "Ele usa um megafone para chamar os filhos ou para reclamar de coisas na rua, causando um alvoroço desnecessário.",
    "Ela deixa o aspirador de pó ligado por horas, mesmo sem estar usando, apenas para gerar um ruído constante.",
    "Ele fica assobiando melodias irritantes e repetitivas em volume alto sempre que me vê no elevador ou nas áreas comuns.",
    "Ela tem um pássaro extremamente barulhento na varanda que grita incessantemente, e ela não faz nada para controlar.",
    "Ele fica batucando em baldes ou latas no quintal, como se estivesse ensaiando para uma banda de sucata, em horários inoportunos.",
    "Ela deixa o portão automático da garagem abrindo e fechando repetidamente, sem motivo aparente, causando um barulho irritante.",
    "Ele fica imitando sons de animais ou sirenes em voz alta, só para me provocar e tirar minha paz.",
    "Ela tem o costume de varrer a casa fazendo muito barulho, arrastando os móveis de forma ostensiva, de madrugada.",
    "Ele estaciona o carro com o som ligado no último volume na frente da minha casa e fica lá por horas.",
    "Ela deixa o liquidificador ligado por longos períodos, mesmo sem estar preparando nada, o barulho é enlouquecedor.",
    
    # Rótulo 9 - Lesão corporal
    "Durante a discussão, ele me empurrou tão forte contra a parede que eu caí e bati a cabeça, fiquei com um galo enorme e muita dor.",
    "Ela me arranhou o rosto e os braços todos com as unhas e puxou meu cabelo com tanta força que arrancou um tufo, me deixando cheia de marcas doloridas.",
    "Ele me deu um soco no olho que ficou roxo por mais de uma semana, além de um corte no supercílio que precisou de pontos.",
    "Num acesso de raiva, ela jogou um prato em mim que me atingiu na perna, causando um corte profundo e sangramento.",
    "Meu companheiro apertou meu braço com tanta força que ficaram as marcas dos dedos dele, muito roxas e sensíveis por vários dias.",
    "Levei um chute nas costelas que me deixou sem ar e com uma dor terrível para respirar por quase um mês.",
    "Ela me queimou com um cigarro aceso no braço de propósito, porque eu não quis dar dinheiro a ela.",
    "Ele torceu meu pulso com violência quando tentei pegar meu celular de volta, e agora meu pulso está inchado e não consigo mexer direito.",
    "Fui atingida por um objeto pesado que ele arremessou na minha direção durante uma briga, o que me causou uma contusão feia na testa.",
    "Ele me bateu com uma panela quente na cabeça, causando um corte e uma queimadura.",
    "Ela me arrastou pelos cabelos pela casa toda, me causando dor intensa e perda de fios.",
    "Levei uma joelhada dele no estômago que me fez vomitar e ficar com dores por dias.",
    "Ela me atingiu com um pedaço de pau nas costas, e agora mal consigo me mover.",
    "Ele me deu um 'mata-leão' até eu quase desmaiar, deixando meu pescoço dolorido e marcado.",
    "Fui ferida por ela com estilhaços de um copo que ela quebrou e jogou em minha direção.",
    "Ele me trancou para fora de casa no frio e jogou água gelada em mim, causando hipotermia leve.",
    "Ela me deu um choque com um aparelho elétrico (taser de defesa pessoal) durante uma briga.",
    "Fui atingido por ele com uma cadeira, que quebrou no impacto e me deixou com vários hematomas.",
    "Ela me envenenou com uma substância na comida que me causou forte intoxicação alimentar e precisei ser hospitalizada.",
    "Ele me cortou com um caco de vidro no braço, resultando em um ferimento que precisou de sutura.",
    "Levei um soco inglês dele nas costelas, que trincou uma delas.",
    "Ela me empurrou escada abaixo, e eu rolei vários degraus, resultando em múltiplas contusões.",
    "Ele me asfixiou com um travesseiro até eu perder os sentidos por alguns segundos.",
    "Fui atingida por ela com uma pedra na cabeça, o que me causou um traumatismo craniano leve.",
    "Ele me deu um golpe de artes marciais no pescoço que me deixou paralisada momentaneamente.",
    "Ela me furou com um garfo na mão durante o jantar, após uma discussão banal.",
    "Levei vários tapas e socos dele no rosto e na cabeça, fiquei completamente atordoada.",
    "Ele me chicoteou com um fio elétrico nas pernas, deixando marcas profundas.",
    "Ela me jogou contra uma estante de vidro, que quebrou e me causou diversos cortes pelo corpo.",
    
    # Rótulo 10 - Tentativa de feminicídio
    "Ele me esfaqueou várias vezes no pescoço e no peito, gritando que se eu não fosse dele não seria de mais ninguém. Só parou porque os vizinhos ouviram meus gritos e chamaram a polícia a tempo.",
    "Ele tentou me estrangular com um fio elétrico na frente das crianças, dizendo que ia acabar com a minha vida. Só não conseguiu porque meu filho mais velho o mordeu e eu consegui gritar por socorro.",
    "Depois que eu disse que ia embora de vez, ele jogou álcool em mim e pegou um isqueiro, ameaçando me queimar viva. Fui salva pela intervenção rápida da minha irmã que chegou na hora.",
    "Ele atirou na minha direção duas vezes com um revólver dentro de casa. Por sorte, nenhum dos tiros me atingiu fatalmente, mas um pegou de raspão no braço. Ele só parou porque a arma travou.",
    "Meu ex-marido me empurrou da escada com clara intenção de me matar, pois eu caí de uma altura considerável. Fraturei a bacia e só não morri por um milagre.",
    "Ele me deu vários golpes na cabeça com uma barra de ferro, repetindo que ia me matar para eu 'aprender a respeitá-lo'. Desmaiei e acordei no hospital, ele fugiu achando que eu estava morta.",
    "Durante uma crise de ciúmes, ele me afogou na banheira e eu perdi a consciência. Acredito que ele pensou que eu tinha morrido, pois quando acordei ele não estava mais lá. Fugi e chamei a polícia.",
    "Ele me deu veneno misturado na comida, dizendo que era um 'remédio especial'. Comecei a passar muito mal e só fui salva porque consegui ligar para uma amiga que me levou às pressas para o hospital.",
    "Ele me injetou uma substância desconhecida à força, dizendo que era para eu 'dormir para sempre'. Consegui chamar uma ambulância antes de perder a consciência.",
    "Durante uma discussão, ele pegou uma espingarda carregada, apontou para mim e apertou o gatilho. A arma falhou, e eu fugi.",
    "Ele me jogou de uma ponte em um rio caudaloso, gritando que eu 'não ia escapar dessa vez'. Fui arrastada pela correnteza mas consegui me agarrar a um galho.",
    "Meu ex-marido me enterrou viva em um buraco no quintal, mas fui encontrada por um vizinho que ouviu meus gritos abafados.",
    "Ele me deu vários choques elétricos com fios desencapados em partes vitais do corpo, e só parou quando achou que eu estava morta.",
    "Ele me prendeu em um quarto e colocou fogo na casa, me deixando para morrer queimada. Consegui sair por uma janela pequena.",
    "Durante uma briga, ele me atingiu na cabeça com uma marreta, e continuei sendo agredida mesmo depois de cair. Fingi-me de morta para ele parar.",
    "Ele me forçou a engolir uma grande quantidade de comprimidos tarja preta, dizendo que era 'o fim da linha' para mim. Vomitei a tempo de ser socorrida.",
    "Ele me atacou com um facão, desferindo golpes no meu pescoço e braços, e só parou quando um segurança do prédio interveio.",
    "Meu parceiro serrou a árvore do quintal para que ela caísse sobre a parte da casa onde eu estava dormindo. Acordei com o estrondo e escapei por pouco.",
    "Ele me prendeu na mala do carro e dirigiu em alta velocidade, ameaçando jogar o carro de um penhasco comigo dentro.",
    "Durante uma discussão, ele me empurrou em direção a uma fogueira acesa, tentando me queimar. Consegui desviar no último segundo.",
    "Ele me deu um golpe de 'gravata' por trás e me arrastou para a piscina, tentando me afogar. Lutei e consegui escapar.",
    "Meu ex tentou me enforcar com o cinto de segurança do carro enquanto dirigia, só parou quando quase batemos.",
    "Ele me deu uma paulada na nuca e, quando caí, continuou a me chutar na cabeça, dizendo que ia 'me apagar'.",
    "Ele me serviu uma bebida com uma substância que paralisou meus músculos e depois tentou me sufocar. Recuperei os movimentos a tempo de reagir.",
    "Durante uma briga, ele pegou uma serra elétrica e a ligou, vindo em minha direção e dizendo que ia 'me picar em pedacinhos'.",
    "Ele me trancou em um freezer industrial desligado, mas que estava em um local isolado, esperando que eu morresse de hipotermia ou asfixia.",
    "Meu ex-companheiro me atacou com uma besta (arma de flechas), e uma flecha atingiu meu ombro perto do pescoço. Ele foi impedido de atirar novamente por terceiros.",
    "Ele adulterou o gás do fogão para causar uma explosão quando eu fosse cozinhar. Senti o cheiro forte e chamei os bombeiros.",

    # Rótulo 11 - Estupro
    "Ele me forçou a ter relações sexuais com ele mesmo eu dizendo 'não' claramente várias vezes e chorando muito. Usou a força física para me imobilizar na cama.",
    "Depois de me embebedar à força em uma festa, ele praticou atos libidinosos comigo enquanto eu estava desacordada e incapaz de consentir.",
    "Meu marido me obrigou a manter conjunção carnal com ele sob ameaça de machucar nossos filhos se eu não cedesse à sua vontade.",
    "Fui estuprada por ele no carro. Ele me ameaçou com uma faca para que eu não reagisse e fizesse tudo o que ele queria.",
    "Ele se aproveitou de um momento em que eu estava dormindo e vulnerável para praticar atos sexuais comigo sem meu consentimento.",
    "Mesmo eu pedindo para parar e tentando afastá-lo, ele continuou a me tocar de forma invasiva e a praticar atos libidinosos contra a minha vontade.",
    "Sob grave ameaça psicológica, dizendo que divulgaria fotos íntimas minhas, ele me constrangeu a ter relações sexuais com ele.",
    "Ele usou sua superioridade física para me subjugar e me violentar sexualmente após uma discussão em que eu disse que queria o divórcio.",
    "Ele me chantageou, dizendo que só me daria o remédio que eu precisava se eu fizesse sexo com ele.",
    "Fui abusada sexualmente por ele enquanto estava sob o efeito de medicamentos fortes que me deixaram sonolenta e sem reação.",
    "Ele me constrangeu a praticar atos libidinosos com ele em troca de não me denunciar por uma infração que cometi.",
    "Meu ex-parceiro me estuprou como forma de 'punição' por eu ter saído com minhas amigas sem a permissão dele.",
    "Ele introduziu objetos em mim sem meu consentimento, causando dor e humilhação, após me imobilizar.",
    "Fui forçada a me despir na frente dele e de seus amigos, sob ameaça de agressão física, e depois fui tocada por eles.",
    "Ele me ameaçou de contar para minha família que sou homossexual se eu não tivesse relações com ele.",
    "Durante uma sessão de massagem que deveria ser terapêutica, ele começou a me tocar de forma sexualizada e me estuprou.",
    "Ele se aproveitou de um momento de fragilidade minha, após a morte de um parente, para me forçar a ter relações.",
    "Fui estuprada corretivamente por ele, que disse que ia 'me ensinar a ser mulher de verdade'.",
    "Ele me filmou secretamente durante o banho e depois usou essas imagens para me chantagear e me forçar a praticar atos sexuais.",
    "Meu chefe me assediou por meses e, um dia, me trancou na sala dele e me estuprou, ameaçando acabar com minha carreira.",
    "Ele me obrigou a participar de um 'ménage à trois' com ele e outra pessoa, contra a minha vontade e sob coação.",
    "Fui forçada a ingerir drogas que me deixaram desorientada e ele se aproveitou desse estado para me violentar.",
    "Ele usou sua posição de líder religioso para me manipular e me constranger a ter relações sexuais, dizendo que era 'vontade divina'.",
    "Fui abusada por ele durante uma consulta médica, onde ele realizou toques íntimos desnecessários e libidinosos.",
    "Ele me estuprou e depois me ameaçou para que eu tomasse a pílula do dia seguinte, para apagar as evidências.",
    "Meu padrasto me estuprava desde a infância, ameaçando matar minha mãe se eu contasse a alguém.",
    "Ele me forçou a ter relações anais, mesmo eu gritando de dor e pedindo para ele parar.",
    "Fui estuprada por um grupo de homens amigos do meu ex-marido, que assistiu a tudo e incentivou.",

    # Rótulo 12 - Sequestro
    "Ele me pegou à força na saída do meu curso, me colocou dentro do carro dele e me levou para um motel na beira da estrada, onde me manteve presa por quase um dia inteiro, tomando meu celular.",
    "Minha ex-namorada não aceitou o término e, quando fui buscar minhas coisas na casa dela, ela me trancou lá, tirou meu celular e disse que eu só sairia se reatasse com ela. Fiquei lá por horas, apavorada.",
    "Ele me abordou na rua, me obrigou a entrar no veículo dele sob ameaça e dirigiu por horas sem rumo certo, me interrogando e me impedindo de sair ou pedir ajuda.",
    "Fui levada por ele para uma casa de praia abandonada e mantida lá contra minha vontade por todo o fim de semana, ele dizia que era para a gente 'se acertar'.",
    "Ela me atraiu para um encontro com a desculpa de conversar e, chegando lá, me levou para um local ermo, tomou minhas chaves e celular, e me impediu de ir embora por várias horas.",
    "Quando anunciei que ia visitar minha família em outra cidade, ele escondeu minhas malas, me trancou no apartamento e me impediu de viajar, me mantendo sob vigilância constante.",
    "Ele me arrastou para dentro do carro e me levou para a casa dos pais dele no interior, me mantendo lá por dias, dizendo que só me traria de volta se eu prometesse obedecê-lo.",
    "Ele me forçou a entrar em um táxi com ele e me levou para um endereço desconhecido, onde me manteve refém até eu concordar em não denunciá-lo.",
    "Minha ex me atraiu para um suposto passeio de lancha e, em alto mar, desligou o motor e disse que só voltaríamos se eu aceitasse suas condições.",
    "Ele me surpreendeu no estacionamento do shopping, me colocou à força no porta-malas do carro e dirigiu por horas, me deixando em pânico.",
    "Fui levada por ela para uma comunidade isolada no interior, sem meu consentimento, e lá fui mantida sob vigilância constante pelos parentes dela.",
    "Ele me dopou em um bar e, quando acordei, estava em um avião particular, sendo levada para outro país contra a minha vontade.",
    "Ela me convenceu a fazer uma viagem de carro 'sem destino' e, quando percebi que estávamos muito longe e quis voltar, ela se recusou e me manteve na estrada por dias.",
    "Ele me sequestrou na saída de uma festa e me levou para uma cabana na floresta, onde me estuprou e me manteve amarrada.",
    "Fui atraído para uma emboscada por ela e seus comparsas, que me levaram para um cativeiro e exigiram resgate da minha família.",
    "Ele me forçou a embarcar em um navio cargueiro, onde fui mantida presa em um contêiner por vários dias, como forma de 'me ensinar uma lição'.",
    "Minha ex-parceira, com ajuda de amigos, me pegou na rua, me amordaçou e me levou para uma casa abandonada, onde fui torturada psicologicamente.",
    "Ele me sequestrou e me levou para o deserto, me deixando lá sem água e sem comida, como vingança por eu ter terminado o relacionamento.",
    "Fui levada à força para um culto religioso extremista em outra cidade e lá fui submetida a um 'ritual de purificação' contra minha vontade.",
    "Ele me prendeu em um bunker subterrâneo que ele construiu, com a intenção de me manter lá indefinidamente como sua 'propriedade'.",
    "Ela me atraiu para um hotel fazenda com promessas de reconciliação e lá me manteve incomunicável, controlando todos os meus passos e contatos.",
    "Ele me sequestrou da porta da minha casa e me levou para uma ilha particular, onde me manteve como sua prisioneira sexual.",
    "Fui forçada a entrar em um caminhão baú por ele e seus cúmplices, e transportada para outro estado, onde fui obrigada a trabalhar em regime de escravidão.",
    "Ela me pegou na saída da escola do meu filho e me levou para um local desconhecido, ameaçando a vida dele se eu não colaborasse.",
    "Ele me manteve refém em um avião que ele mesmo pilotava, ameaçando derrubar a aeronave se eu não reatasse o namoro.",
    "Fui levada para uma clínica psiquiátrica particular contra minha vontade por ele, que alegou que eu estava louca, para se livrar de mim.",
    "Ela me sequestrou e me levou para um país onde eu não conhecia ninguém e não falava a língua, me deixando completamente dependente dela."
]

# -----------------------------------------------------------------------------

# Inicializando a lista de rótulos de forma automatizada
labels = []

# Adicionando 30 rótulos para cada tipo de violência
labels.extend([0] * 39)
labels.extend([1] * 38)
labels.extend([2] * 39)
labels.extend([3] * 39)
labels.extend([4] * 27)
labels.extend([5] * 27)
labels.extend([6] * 28)
labels.extend([7] * 28)
labels.extend([8] * 29)
labels.extend([9] * 28)
labels.extend([10] * 28)
labels.extend([11] * 27)

# Verificando o total de dados gerados
print(f"Total de textos gerados: {len(data)}")
print(f"Total de rótulos gerados: {len(labels)}")

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

# Avaliar o modelo
y_pred = clf.predict(X_test_encoded)
print(classification_report(y_test, y_pred, target_names
                            = class_names))

# Função de predição
def predict(text):
    # Transformar o novo dado em embedding
    text_encoded = encode_texts(pd.Series([text]))
    prediction = clf.predict(text_encoded)
    return prediction[0]


@app.route('/classify', methods=['POST'])
def classify_text():
    class_names = [
    "Ameaça por violência doméstica ou familiar",      # Rótulo 1
    "Calúnia por violência doméstica ou familiar",     # Rótulo 2
    "Difamação por violência doméstica ou familiar",   # Rótulo 3
    "Injúria por violência doméstica ou familiar",     # Rótulo 4
    "Constrangimento ilegal por violência doméstica ou familiar", # Rótulo 5
    "Cárcere privado por violência doméstica ou familiar",     # Rótulo 6
    "Descumprimento de medida protetiva de urgência",  # Rótulo 7
    "Perturbação do sossego por violência doméstica ou familiar", # Rótulo 8
    "Lesão corporal",                                  # Rótulo 9
    "Tentativa de feminicídio",                        # Rótulo 10
    "Estupro",                                         # Rótulo 11
    "Sequestro"                                        # Rótulo 12
]
    data = request.get_json()
    text = data['text']

    # Predição
    prediction = predict(text)
    
    # Retornar a classificação   
    return jsonify({"result": class_names[prediction]})

if __name__ == '__main__':
    app.run(debug=True)