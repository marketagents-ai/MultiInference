import asyncio
from dotenv import load_dotenv
from minference.lite.inference import InferenceOrchestrator, RequestLimits
from minference.lite.models import Usage,ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool
from typing import Literal, List
from minference.entity import EntityRegistry
from minference.caregistry import CallableRegistry
import time
import os

async def main():
    load_dotenv()
    EntityRegistry()
    CallableRegistry()
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_model = "openai/NousResearch/Hermes-3-Llama-3.1-8B"




    orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits, litellm_request_limits=lite_llm_request_limits)

    json_schema = {
        "type": "object",
        "properties": {
            "english_translation": {"type": "string"},
            "figures_of_speech": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "figure_type": {"type": "string"},
                        "text": {"type": "string"},
                        "explanation": {"type": "string"},
                        "line_numbers": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["figure_type", "text", "explanation", "line_numbers"]
                }
            }
        },
        "required": ["english_translation", "figures_of_speech"],
        "additionalProperties": False
    }

    structured_tool = StructuredTool(
        json_schema=json_schema,
        name="analyze_dante",
        description="Translate the given text to English and identify figures of speech"
    )
    system_string = SystemPrompt(
        content="You are a literary expert specializing in Dante's Divine Comedy. Your task is to provide accurate English translations and identify figures of speech in the text.",
        name="dante_analyzer"
    )

    def create_chats(client:LLMClient, model, response_formats : List[ResponseFormat]= [ResponseFormat.text], count=1) -> List[ChatThread]:
        chats : List[ChatThread] = []
        for response_format in response_formats:
            llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=2000)
            for i in range(count):
                chats.append(
                    ChatThread(
                    system_prompt=system_string,
                    new_message="""Please translate the following excerpt from Dante's Inferno to English  and identify all figures of speech, including metaphors, similes, personification, and other literary devices. Provide line numbers for each figure of speech identified. 
                    The excerpt is Già era in loco onde s’udìa ’l rimbombo 
de l’acqua che cadea ne l’altro giro, 
simile a quel che l’arnie fanno rombo,                           3

quando tre ombre insieme si partiro, 
correndo, d’una torma che passava 
sotto la pioggia de l’aspro martiro.                                 6

Venian ver noi, e ciascuna gridava: 
«Sòstati tu ch’a l’abito ne sembri 
esser alcun di nostra terra prava».                                 9

Ahimè, che piaghe vidi ne’ lor membri 
ricenti e vecchie, da le fiamme incese! 
Ancor men duol pur ch’i’ me ne rimembri.                   12

A le lor grida il mio dottor s’attese; 
volse ’l viso ver me, e: «Or aspetta», 
disse «a costor si vuole esser cortese.                        15

E se non fosse il foco che saetta 
la natura del loco, i’ dicerei 
che meglio stesse a te che a lor la fretta».                   18

Ricominciar, come noi restammo, ei 
l’antico verso; e quando a noi fuor giunti, 
fenno una rota di sé tutti e trei.                                        21

Qual sogliono i campion far nudi e unti, 
avvisando lor presa e lor vantaggio, 
prima che sien tra lor battuti e punti,                             24

così, rotando, ciascuno il visaggio 
drizzava a me, sì che ’n contraro il collo 
faceva ai piè continuo viaggio.                                        27

E «Se miseria d’esto loco sollo 
rende in dispetto noi e nostri prieghi», 
cominciò l’uno «e ’l tinto aspetto e brollo,                    30

la fama nostra il tuo animo pieghi 
a dirne chi tu se’, che i vivi piedi 
così sicuro per lo ’nferno freghi.                                     33

Questi, l’orme di cui pestar mi vedi, 
tutto che nudo e dipelato vada, 
fu di grado maggior che tu non credi:                            36

nepote fu de la buona Gualdrada; 
Guido Guerra ebbe nome, e in sua vita 
fece col senno assai e con la spada.                           39

L’altro, ch’appresso me la rena trita, 
è Tegghiaio Aldobrandi, la cui voce 
nel mondo sù dovrìa esser gradita.                               42

E io, che posto son con loro in croce, 
Iacopo Rusticucci fui; e certo 
la fiera moglie più ch’altro mi nuoce».                          45

S’i’ fossi stato dal foco coperto, 
gittato mi sarei tra lor di sotto, 
e credo che ’l dottor l’avrìa sofferto;                                48

ma perch’io mi sarei brusciato e cotto, 
vinse paura la mia buona voglia 
che di loro abbracciar mi facea ghiotto.                         51

Poi cominciai: «Non dispetto, ma doglia 
la vostra condizion dentro mi fisse, 
tanta che tardi tutta si dispoglia,                                     54

tosto che questo mio segnor mi disse 
parole per le quali i’ mi pensai 
che qual voi siete, tal gente venisse.                             57

Di vostra terra sono, e sempre mai 
l’ovra di voi e li onorati nomi 
con affezion ritrassi e ascoltai.                                        60

Lascio lo fele e vo per dolci pomi 
promessi a me per lo verace duca; 
ma ’nfino al centro pria convien ch’i’ tomi».                 63

«Se lungamente l’anima conduca 
le membra tue», rispuose quelli ancora, 
«e se la fama tua dopo te luca,                                       66

cortesia e valor dì se dimora 
ne la nostra città sì come suole, 
o se del tutto se n’è gita fora;                                          69

ché Guiglielmo Borsiere, il qual si duole 
con noi per poco e va là coi compagni, 
assai ne cruccia con le sue parole».                            72

«La gente nuova e i sùbiti guadagni 
orgoglio e dismisura han generata, 
Fiorenza, in te, sì che tu già ten piagni».                       75

Così gridai con la faccia levata; 
e i tre, che ciò inteser per risposta, 
guardar l’un l’altro com’al ver si guata.                         78

«Se l’altre volte sì poco ti costa», 
rispuoser tutti «il satisfare altrui, 
felice te se sì parli a tua posta!                                        81

Però, se campi d’esti luoghi bui 
e torni a riveder le belle stelle, 
quando ti gioverà dicere "I’ fui",                                       84

fa che di noi a la gente favelle». 
Indi rupper la rota, e a fuggirsi 
ali sembiar le gambe loro isnelle.                                 87

Un amen non saria potuto dirsi 
tosto così com’e’ fuoro spariti; 
per ch’al maestro parve di partirsi.                                 90

Io lo seguiva, e poco eravam iti, 
che ’l suon de l’acqua n’era sì vicino, 
che per parlar saremmo a pena uditi.                           93

Come quel fiume c’ha proprio cammino 
prima dal Monte Viso ’nver’ levante, 
da la sinistra costa d’Apennino,                                     96

che si chiama Acquacheta suso, avante 
che si divalli giù nel basso letto, 
e a Forlì di quel nome è vacante,                                    99

rimbomba là sovra San Benedetto 
de l’Alpe per cadere ad una scesa 
ove dovea per mille esser recetto;                                102

così, giù d’una ripa discoscesa, 
trovammo risonar quell’acqua tinta, 
sì che ’n poc’ora avria l’orecchia offesa.                      105

Io avea una corda intorno cinta, 
e con essa pensai alcuna volta 
prender la lonza a la pelle dipinta.                                108

Poscia ch’io l’ebbi tutta da me sciolta, 
sì come ’l duca m’avea comandato, 
porsila a lui aggroppata e ravvolta.                               111

Ond’ei si volse inver’ lo destro lato, 
e alquanto di lunge da la sponda 
la gittò giuso in quell’alto burrato.                                 114

’E’ pur convien che novità risponda’ 
dicea fra me medesmo, ’al novo cenno 
che ’l maestro con l’occhio sì seconda’.                     117

Ahi quanto cauti li uomini esser dienno 
presso a color che non veggion pur l’ovra, 
ma per entro i pensier miran col senno!                     120

El disse a me: «Tosto verrà di sovra 
ciò ch’io attendo e che il tuo pensier sogna: 
tosto convien ch’al tuo viso si scovra».                        123

Sempre a quel ver c’ha faccia di menzogna 
de’ l’uom chiuder le labbra fin ch’el puote, 
però che sanza colpa fa vergogna;                               126

ma qui tacer nol posso; e per le note 
di questa comedìa, lettor, ti giuro, 
s’elle non sien di lunga grazia vòte,                             129

ch’i’ vidi per quell’aere grosso e scuro 
venir notando una figura in suso, 
maravigliosa ad ogne cor sicuro,                                 132

sì come torna colui che va giuso 
talora a solver l’àncora ch’aggrappa 
o scoglio o altro che nel mare è chiuso, 

che ’n sù si stende, e da piè si rattrappa.                   136:
                    """,
                    llm_config=llm_config,
                    forced_output=structured_tool,
                )

            )
        return chats

    # OpenAI chats
    openai_chats = create_chats(LLMClient.openai, "gpt-4o-mini", [ResponseFormat.tool], 5)
    litellm_chats = create_chats(LLMClient.litellm, lite_llm_model, [ResponseFormat.tool], 5)
    
    all_chats = openai_chats + litellm_chats
    all_chats = litellm_chats
    chats_id = [chat.id for chat in all_chats]
        

    # print(chats[0].llm_config)
    print("Running parallel completions...")

    start_time = time.time()
    # with Session(engine) as session:
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    for chat in all_chats:

        chat.new_message = "And why is it funny?"
    second_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    num_text = 0
    num_json = 0
    total_calls = 0
    return all_chats


if __name__ == "__main__":
    all_chats = asyncio.run(main())
    print(all_chats[0].get_all_usages())
    print(EntityRegistry.list_by_type(Usage))