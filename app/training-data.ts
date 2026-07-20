export type TrainingPhase = "P1" | "P2" | "Guided";
export type TrackingProfile = "standard" | "sovt" | "quiet" | "agility" | "unvoiced" | "guided";
export type ExerciseKind = "pitch" | "glide" | "sustain" | "rhythm" | "breath" | "guided";

export type TrainingExercise = {
  id: string;
  module: number;
  title: string;
  shortTitle: string;
  syllable: string;
  phase: TrainingPhase;
  profile: TrackingProfile;
  kind: ExerciseKind;
  description: string;
  cue: string;
  scoreLabel: string;
  pattern: number[];
  beats: number[];
  bpm: number;
  startMidi: number;
  guidedSeconds?: number;
};

export const TRAINING_MODULES = [
  { id: 1, title: "Breath management", kicker: "Build a steady, economical airflow", color: "mint" },
  { id: 2, title: "SOVT warm-ups", kicker: "Find easy resonance with less pressure", color: "aqua" },
  { id: 3, title: "Tension & onset", kicker: "Release effort and start notes cleanly", color: "peach" },
  { id: 4, title: "Scales & intonation", kicker: "Center every pitch with confidence", color: "gold" },
  { id: 5, title: "Register & mix", kicker: "Connect chest and head voice smoothly", color: "violet" },
  { id: 6, title: "Agility & runs", kicker: "Make fast notes clean and intentional", color: "coral" },
  { id: 7, title: "Dynamics & control", kicker: "Shape volume without losing the note", color: "blue" },
] as const;

const make = (exercise: TrainingExercise) => exercise;
const MAJOR_ARPEGGIO = [0, 4, 7, 12, 7, 4, 0] as const;

export const TRAINING_EXERCISES: TrainingExercise[] = [
  make({ id:"sustained-hiss", module:1, title:"Sustained hiss", shortTitle:"Sustained hiss", syllable:"S / SH", phase:"P1", profile:"unvoiced", kind:"breath", description:"Exhale for as long and as evenly as you can.", cue:"Keep the stream narrow and consistent—no need to push.", scoreLabel:"Airflow steadiness", pattern:[], beats:[], bpm:60, startMidi:60, guidedSeconds:20 }),
  make({ id:"panting", module:1, title:"Panting drill", shortTitle:"Panting", syllable:"HA", phase:"P1", profile:"unvoiced", kind:"rhythm", description:"Pulse short, unvoiced bursts on the beat.", cue:"Let the abdomen rebound; keep the jaw and throat easy.", scoreLabel:"Pulse regularity", pattern:[0,0,0,0,0,0,0,0], beats:[.5,.5,.5,.5,.5,.5,.5,.5], bpm:80, startMidi:60 }),
  make({ id:"box-breathing", module:1, title:"Box breathing", shortTitle:"Box breathing", syllable:"4 · 4 · 4 · 4", phase:"Guided", profile:"guided", kind:"guided", description:"Follow four equal inhale, suspend, exhale, and rest phases.", cue:"Stay comfortable. Shorten the count if you feel strain or lightheadedness.", scoreLabel:"Cycles complete", pattern:[], beats:[], bpm:60, startMidi:60, guidedSeconds:64 }),
  make({ id:"farinelli", module:1, title:"Extended-exhale progression", shortTitle:"Farinelli", syllable:"SSS", phase:"P1", profile:"unvoiced", kind:"breath", description:"A progressive sustained exhale that grows with your practice history.", cue:"Finish with air in reserve. Consistency matters more than the count.", scoreLabel:"Steady duration", pattern:[], beats:[], bpm:60, startMidi:60, guidedSeconds:24 }),

  make({ id:"humming-glides", module:2, title:"Humming glides", shortTitle:"Humming glides", syllable:"MM", phase:"P1", profile:"standard", kind:"glide", description:"Glide up and down on an easy hum.", cue:"Feel vibration at the lips; connect the full arc without steps.", scoreLabel:"Contour smoothness", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.55), bpm:72, startMidi:55 }),
  make({ id:"lip-trill", module:2, title:"Lip-trill glides & sustains", shortTitle:"Lip trill", syllable:"BRR", phase:"P2", profile:"sovt", kind:"glide", description:"Keep the lips bubbling through a gentle siren.", cue:"Use only enough air to keep the trill moving.", scoreLabel:"Voicing continuity", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.6), bpm:68, startMidi:55 }),
  make({ id:"tongue-trill", module:2, title:"Tongue-trill glides", shortTitle:"Tongue trill", syllable:"RRR", phase:"P2", profile:"sovt", kind:"glide", description:"Roll the tongue continuously across the pitch arc.", cue:"Keep the tongue tip loose and let airflow do the work.", scoreLabel:"Voicing continuity", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.7), bpm:68, startMidi:55 }),
  make({ id:"straw-phonation", module:2, title:"Straw phonation", shortTitle:"Straw phonation", syllable:"OO", phase:"P2", profile:"quiet", kind:"glide", description:"Glide and sustain through a straw with very gentle airflow.", cue:"A quiet sound is correct. Follow the shape, not the loudness.", scoreLabel:"Contour shape", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.8), bpm:62, startMidi:55 }),
  make({ id:"voiced-fricatives", module:2, title:"Voiced fricative slides", shortTitle:"V / Z / ZH", syllable:"VVV", phase:"P2", profile:"sovt", kind:"glide", description:"Slide while keeping the fricative continuously voiced.", cue:"Keep a gentle buzz under the airflow noise.", scoreLabel:"Pitch continuity", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.55), bpm:74, startMidi:57 }),
  make({ id:"blowfish", module:2, title:"Blowfish phonation", shortTitle:"Blowfish", syllable:"BOO", phase:"P2", profile:"quiet", kind:"glide", description:"Cup the cheeks lightly and phonate through a small opening.", cue:"Let the cheeks absorb pressure; avoid clamping the lips.", scoreLabel:"Contour shape", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.7), bpm:66, startMidi:55 }),

  make({ id:"jaw-release", module:3, title:"Jaw-release routine", shortTitle:"Jaw release", syllable:"YAWN", phase:"Guided", profile:"guided", kind:"guided", description:"A guided stretch, massage, and silent-yawn sequence.", cue:"Nothing should hurt. Let the tongue rest forward and breathe normally.", scoreLabel:"Steps complete", pattern:[], beats:[], bpm:60, startMidi:60, guidedSeconds:70 }),
  make({ id:"vowel-morph", module:3, title:"Consonant-vowel morphing", shortTitle:"M–vowel morph", syllable:"M · MAH · MAY · MEE · MOH · MOO", phase:"P1", profile:"standard", kind:"sustain", description:"Hold one pitch while the mouth changes shape.", cue:"Keep the tone connected and the pitch still through each vowel.", scoreLabel:"Pitch stability", pattern:[0,0,0,0,0,0], beats:[1,1,1,1,1,1], bpm:66, startMidi:60 }),
  make({ id:"staccato-onset", module:3, title:"Staccato onset arpeggios", shortTitle:"Clean onsets", syllable:"HA", phase:"P2", profile:"agility", kind:"pitch", description:"Strike each note directly on pitch and on the beat.", cue:"Think small and buoyant—avoid scooping up to the target.", scoreLabel:"Onset accuracy", pattern:[0,4,7,4,0], beats:[1,1,1,1,1], bpm:76, startMidi:60 }),

  make({ id:"five-tone", module:4, title:"Five-tone scale", shortTitle:"Five-tone scale", syllable:"AH", phase:"P1", profile:"standard", kind:"pitch", description:"The foundational 1–2–3–4–5 scale, up and back down.", cue:"Aim for the center of each note and keep the vowels matched.", scoreLabel:"Pitch accuracy", pattern:[0,2,4,5,7,5,4,2,0], beats:Array(9).fill(1), bpm:84, startMidi:60 }),
  make({ id:"octave-jump", module:4, title:"Octave jump", shortTitle:"Octave jump", syllable:"YAH", phase:"P1", profile:"standard", kind:"pitch", description:"Land directly on the octave without sliding into it.", cue:"Hear the upper note first; let it feel tall, not loud.", scoreLabel:"Landing accuracy", pattern:[0,12,0,12,0], beats:[1.5,1.5,1.5,1.5,1.5], bpm:66, startMidi:55 }),
  make({ id:"rossini", module:4, title:"Rossini arpeggio", shortTitle:"Rossini arpeggio", syllable:"AH", phase:"P1", profile:"standard", kind:"pitch", description:"Cross the register on 1–3–5–8–10 and return.", cue:"Keep the tone buoyant as you pass through the middle voice.", scoreLabel:"Note accuracy", pattern:[0,4,7,12,16,12,7,4,0], beats:Array(9).fill(.85), bpm:88, startMidi:53 }),
  make({ id:"chromatic", module:4, title:"Chromatic scale", shortTitle:"Chromatic scale", syllable:"NOO", phase:"P1", profile:"agility", kind:"pitch", description:"Place every semitone evenly without tonal shortcuts.", cue:"Move in tiny, equal steps and listen ahead.", scoreLabel:"Semitone precision", pattern:[0,1,2,3,4,5,6,7,8,9,10,11,12], beats:Array(13).fill(.7), bpm:74, startMidi:55 }),
  make({ id:"whole-tone", module:4, title:"Whole-tone scale", shortTitle:"Whole-tone scale", syllable:"OH", phase:"P1", profile:"standard", kind:"pitch", description:"Build interval independence with equal whole steps.", cue:"Keep every step the same size—there is no leading tone.", scoreLabel:"Interval accuracy", pattern:[0,2,4,6,8,10,12,10,8,6,4,2,0], beats:Array(13).fill(.7), bpm:80, startMidi:55 }),
  make({ id:"sustained-note", module:4, title:"Sustained-note hold", shortTitle:"Sustained note", syllable:"AH", phase:"P1", profile:"standard", kind:"sustain", description:"Hold one centered pitch with an even, free tone.", cue:"Let vibrato happen naturally; avoid steering every wobble.", scoreLabel:"Pitch stability", pattern:[0], beats:[8], bpm:60, startMidi:60 }),

  make({ id:"descending-foo", module:5, title:"Descending Foo / Noo", shortTitle:"Head-first descent", syllable:"FOO", phase:"P1", profile:"standard", kind:"pitch", description:"Begin lightly in head voice and descend through the five-tone scale.", cue:"Carry the easy upper feeling down instead of adding weight.", scoreLabel:"Blend continuity", pattern:[7,5,4,2,0], beats:Array(5).fill(1), bpm:76, startMidi:57 }),
  make({ id:"ney-octave", module:5, title:"Ney octave scale", shortTitle:"Ney octave", syllable:"NEY", phase:"P1", profile:"standard", kind:"pitch", description:"Travel up and over the break with a bright, focused sound.", cue:"Keep it playful and narrow; volume is not the goal.", scoreLabel:"Passaggio accuracy", pattern:[...MAJOR_ARPEGGIO], beats:Array(7).fill(.9), bpm:76, startMidi:55 }),
  make({ id:"yodel-flip", module:5, title:"Yodel / register flip", shortTitle:"Register flip", syllable:"YOH", phase:"P2", profile:"agility", kind:"pitch", description:"Make a quick, deliberate switch between chest and head voice.", cue:"Allow the flip. Do not smooth it into a slide.", scoreLabel:"Flip speed", pattern:[0,9,0,12,0,9,0], beats:Array(7).fill(1), bpm:70, startMidi:55 }),

  make({ id:"three-note-run", module:6, title:"Three-note descending run", shortTitle:"3-note run", syllable:"DA", phase:"P1", profile:"agility", kind:"pitch", description:"Begin staccato, then connect 3–2–1 as the tempo rises.", cue:"Release each note cleanly; speed only follows accuracy.", scoreLabel:"Run accuracy", pattern:[4,2,0,4,2,0,4,2,0], beats:Array(9).fill(.5), bpm:82, startMidi:60 }),
  make({ id:"pentatonic-run", module:6, title:"Five-note pentatonic run", shortTitle:"Pentatonic run", syllable:"YA", phase:"P2", profile:"agility", kind:"pitch", description:"Practice the core 3–2–1–6–5 pop and R&B riff.", cue:"Give every note a clear center; do not blur the turn.", scoreLabel:"Note separation", pattern:[4,2,0,-3,-5,4,2,0,-3,-5], beats:Array(10).fill(.45), bpm:78, startMidi:62 }),
  make({ id:"melisma", module:6, title:"Extended melisma patterns", shortTitle:"Melisma cascade", syllable:"OH", phase:"P2", profile:"agility", kind:"pitch", description:"Link longer pentatonic cascades, then listen back to the take.", cue:"Think in small three-note cells rather than one long run.", scoreLabel:"Melisma clarity", pattern:[7,4,2,0,-3,-5,-8,-5,-3,0,2,4,2,0,-3,-5], beats:Array(16).fill(.4), bpm:76, startMidi:60 }),

  make({ id:"messa-di-voce", module:7, title:"Messa di voce", shortTitle:"Messa di voce", syllable:"AH", phase:"P2", profile:"standard", kind:"sustain", description:"Grow from soft to full and return while the pitch stays centered.", cue:"Shape one smooth arc. Never chase loudness or strain at the peak.", scoreLabel:"Dynamic control", pattern:[0], beats:[12], bpm:60, startMidi:60 }),
];

export function exercisesForModule(moduleId: number) {
  return TRAINING_EXERCISES.filter((exercise) => exercise.module === moduleId);
}
