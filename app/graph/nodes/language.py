from lingua import Language, LanguageDetectorBuilder


languages=[
    Language.ENGLISH,
    Language.HINDI,
    Language.TELUGU
]

detector= (
    LanguageDetectorBuilder
    .from_languages(*languages)
    .build()
)


def detect_language(state):

    text= state['current_transcript']

    detected= detector.detect_language_of(text)

    language_map={
        Language.ENGLISH: "en",
        Language.HINDI: 'hi',
        Language.TELUGU: "te",
    }

    return {
        "lead":{
            "language": language_map.get(detected, 'en')
        }
    }
