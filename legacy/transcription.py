import pretty_midi
from pretty_midi import note_number_to_name, note_name_to_number
import itertools

# Encoding and decoding from pretty_midi object to string

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

# Set boundaries of midi velocity with certain dynamics (loudness) of a piece / note
dyn_vel = {
    "pppp": 10,
    "ppp": 23,
    "pp": 36,
    "p": 49,
    "mp": 62,
    "mf": 75,
    "f": 88,
    "ff": 101,
    "fff": 114,
    "ffff": 127
}

# Inverse of dyn_vel
vel_dyn = dict()
for key in dyn_vel.keys():
    vel_dyn[dyn_vel[key]] = key


# Translate midi note number to (note in octave number, number of the octave)
def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    if not octave in OCTAVES:
        raise Exception()
    if not (0 <= number <= 127):
        raise Exception()
    note = NOTES[number % NOTES_IN_OCTAVE]
    return note, octave


# Inverse of number_to_note(number: int) -> tuple
def note_to_number(note_octave: list) -> int:
    note, octave = note_octave
    if not note in NOTES:
        raise Exception()
    if not octave in OCTAVES:
        raise Exception()

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    if not (0 <= note <= 127):
        raise Exception()

    return note


# Translate midi velocity to dynamics denotation
def velocity_to_dynamics(velocity : int) -> str:
    dynamics = ""
    for key in list(sorted(vel_dyn.keys())):
        if velocity <= key:
            dynamics = vel_dyn[key]
            break
    return dynamics


# Inverse of midi velocity mapping
def dynamics_to_velocity(dynamics : str) -> int:
    velocity = dyn_vel[dynamics]
    return velocity


# Determine time in sixteenth notes (semiquavers) since the start of piece
# of the second for given midi object
def time_to_16th(sec, pm):
    resolution = pm.resolution
    ticks = pm.time_to_tick(sec)
    return int(round(ticks / (resolution * 0.25)))


# Inverse of time_to_16th
def sixteenth_to_time(length, pm):
    resolution = pm.resolution
    ticks = int(length * resolution * 0.25)
    return pm.tick_to_time(ticks)


# Make dict of N empty lists corresponding each time step (semiquaver) in the piece
def get_intervals(pm):
    last_start = max([n.start for n in itertools.chain.from_iterable(i.notes for i in pm.instruments)])
    last_start = time_to_16th(last_start, pm)
    intervals = dict()
    for i in range(last_start + 1):
        intervals[i] = []
    return intervals


# Fill intervals dict with a symbolic description of notes  played
# at corresponding time steps
def fill_intervals(pm):
    intervals = get_intervals(pm)
    res = pm.resolution
    for count, instrument in enumerate(pm.instruments):
        for note in instrument.notes:
            note_name = note_number_to_name(note.pitch)

            start = time_to_16th(note.start, pm)
            length = time_to_16th(note.end - note.start, pm)

            dynamics = velocity_to_dynamics(note.velocity)
            as_string = " ".join([  # "inst_" + str(instrument.program),
                note_name, str(length) + "/16th", dynamics])
            intervals[start].append((count, as_string))
    return intervals


# Concatenate intervals to solid musical piece transcription
def intervals_to_composition(intervals, pm):
    composition = []
    for key in intervals.keys():
        moment = []
        intervals[key].sort(key=lambda x: x[0])

        if len(intervals[key]) == 0:
            composition.append("VOID")
        else:
            for count, instr in enumerate(pm.instruments):
                inst_name = "inst_" + str(instr.program);
                inst_notes = [val[1] for val in intervals[key] if val[0] == count]

                if len(inst_notes) == 0:
                    moment.append(" ")
                else:
                    moment.append(inst_name + " " + " ".join(inst_notes))
            composition.append(" , ".join(moment))

    composition = " ; ".join(composition)
    return " ".join(composition.split())


# Make pretty midi notes from transcripted notes
def get_notes(tick, notes, pm):
    notes = notes.split()[1::]
    notes_processed = []
    for i in range(len(notes) // 3):
        note_name = notes[i * 3]
        note_leng = notes[i * 3 + 1]
        note_dynamics = notes[i * 3 + 2]

        velocity = dynamics_to_velocity(note_dynamics)
        pitch = note_name_to_number(note_name)
        start = sixteenth_to_time(tick, pm)
        end = start + sixteenth_to_time(int(note_leng[:-5:]), pm)

        note = pretty_midi.Note(velocity, pitch, start, end)
        notes_processed.append(note)
    return notes_processed


# Pretty midi object to (transcription, resolution, tempo)
def encode(pm):
    res = pm.resolution
    tempo = pm.estimate_tempo()
    intervals = fill_intervals(pm)
    return intervals_to_composition(intervals, pm), res, tempo


# Pretty midi object of (transcription, resolution, tempo)
def decode(encoded, og_res, og_tempo):
    intervals = dict()
    instruments_info = set()
    instruments = dict()

    for tick, value in enumerate(encoded.split(';')):
        if value.strip() == "VOID":
            intervals[tick] = []
        else:
            moment = list([(i, v.strip()) for i, v in enumerate(value.split(',')) if len(v.strip().split(' ')) > 1])
            intervals[tick] = moment
            for order, inst in moment:
                name = inst.split(' ')[0]
                program = int(name[5::1])
                instruments_info.add((order, name, program))

    instruments_info = list(instruments_info)
    instruments_info.sort(key=lambda x: x[0])

    for order, name, program in instruments_info:
        inst = pretty_midi.Instrument(program, is_drum=False, name=name)
        instruments[name] = inst

    new_pm = pretty_midi.PrettyMIDI(midi_file=None, resolution=og_res, initial_tempo=og_tempo)

    for tick in range(len(intervals.keys())):
        for track in intervals[tick]:
            instrument_name = instruments_info[track[0]][1]
            notes = get_notes(tick, track[1], new_pm)
            instruments[instrument_name].notes += notes

    for _, name, _ in instruments_info:
        new_pm.instruments.append(instruments[name])

    return new_pm
