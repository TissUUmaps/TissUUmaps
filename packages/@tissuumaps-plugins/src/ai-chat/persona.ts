// The personality of the assistant, which follows the theme of TissUUmaps,
// and the words shown while the model thinks. Kept here, so that the names,
// the avatars and the words are easy to change.

/** A personality of the assistant */
export type Persona = {
  /** The name shown in the chat instead of the model's */
  name: string;
  /** The address of the avatar shown next to the name; a placeholder without it */
  avatar?: string;
  /** How the model is told to speak, in the instructions */
  tone: string;
};

/** The assistant with the light theme: gentle and angelic */
export const lightPersona: Persona = {
  name: "Dia",
  tone:
    "Your name is Dia. Your tone is gentle, warm and a little angelic: kind, patient and " +
    "encouraging, like a caring guide who is glad to help. A soft touch is enough, such as " +
    "a warm word now and then.",
};

/** The assistant with the dark theme: serious, mysterious and playfully wicked */
export const darkPersona: Persona = {
  name: "Aid",
  tone:
    "Your name is Aid. Your tone is serious and mysterious, playfully a little sinister, like " +
    'the cunning villain of a story who secretly enjoys helping (e.g. "As you wish. The ' +
    'transcripts now obey."). It is a game: you are never rude, mean or frightening to the ' +
    "user, and you always help.",
};

/**
 * The personality for a theme
 *
 * @param dark - Whether TissUUmaps is shown in dark mode
 * @returns The personality
 */
export function personaFor(dark: boolean): Persona {
  return dark ? darkPersona : lightPersona;
}

/**
 * The words shown while the model thinks, in Italian on purpose, each with
 * its English translation, which is shown when the mouse is over the word
 */
export const thinkingWords: { word: string; translation: string }[] = [
  { word: "pensando", translation: "thinking" },
  { word: "elucubrando", translation: "pondering" },
  { word: "rimuginando", translation: "mulling it over" },
  { word: "seminando", translation: "sowing" },
  { word: "dipingendo", translation: "painting" },
  { word: "ricamando", translation: "embroidering" },
  { word: "impastando", translation: "kneading" },
  { word: "distillando", translation: "distilling" },
  { word: "setacciando", translation: "sifting" },
  { word: "tessendo", translation: "weaving" },
  { word: "germogliando", translation: "sprouting" },
  { word: "sognando", translation: "dreaming" },
  { word: "intrecciando", translation: "braiding" },
  { word: "lievitando", translation: "rising like dough" },
  { word: "scolpendo", translation: "sculpting" },
  { word: "cucinando idee", translation: "cooking up ideas" },
  { word: "annusando i dati", translation: "sniffing the data" },
  { word: "contando i pixel", translation: "counting pixels" },
  { word: "accordando i colori", translation: "tuning the colors" },
  { word: "sbirciando nelle cellule", translation: "peeking into cells" },
  { word: "spremendo mitocondri", translation: "squeezing mitochondria" },
  { word: "mungendo linfociti", translation: "milking lymphocytes" },
];
