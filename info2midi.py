from mido import Message, MidiFile, MidiTrack, MetaMessage

def create_midi(note_types, pitches, output_file="output.mid", tempo=500000):
    """
    Create a MIDI file from note types and pitches.

    Parameters:
    - note_types: List of note types (e.g., clef, bar, #, 4).
    - pitches: List of corresponding pitches for notes (e.g., ['E4', 'F#4', ...]).
    - output_file: Path to save the generated MIDI file.
    - tempo: MIDI tempo in microseconds per beat (default: 120 bpm -> 500000 us/beat).
    """
    # MIDI note mappings
    note_to_midi = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
        'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    def pitch_to_midi(pitch):
        if pitch == 'p':  # Rest
            return None
        note, octave = pitch[:-1], int(pitch[-1])
        return note_to_midi[note] + (octave + 1) * 12

    # Create MIDI file and track
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Set tempo and time signature (default: 4/4)
    track.append(MetaMessage('set_tempo', tempo=tempo))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))

    # Parse note types and pitches
    tick_duration = 480  # Quarter note duration in ticks
    current_time_signature = (3, 4)  # Default time signature

    note_idx = 0
    for note_type in note_types:
        if note_type == 'clef':
            continue  # Skip clef
        elif note_type == 'bar':
            continue  # Skip bar line
        elif note_type.startswith('t'):  # Time signature
            ts = note_type[1:]
            numerator, denominator = int(ts[0]), int(ts[1])
            current_time_signature = (numerator, denominator)
            track.append(MetaMessage('time_signature', numerator=numerator, denominator=denominator))
        elif note_type == '#':  # Sharp symbol (skip for now, as pitch already includes sharp notes)
            continue
        elif note_type.isdigit():  # Quarter note (or other note duration)
            midi_note = pitch_to_midi(pitches[note_idx])
            note_idx += 1
            if midi_note is not None:
                # Add note-on and note-off events
                track.append(Message('note_on', note=midi_note, velocity=64, time=0))
                track.append(Message('note_off', note=midi_note, velocity=64, time=tick_duration))

    # Save the MIDI file
    midi.save(output_file)
    print(f"MIDI file saved as {output_file}")

# Example usage
note_types = ['clef', 'bar', '#', '#', '#', '#', 't34', '4', '4', '4', 'bar', '4', '4', '4', 'bar', '4', '4', '4', 'bar', '4', '4', '4', 'bar']
pitches = ['E4', 'F#4', 'G#4', 'F#4', 'G#4', 'A4', 'G#4', 'A4', 'B4', 'A4', 'B4', 'C#5']

create_midi(note_types, pitches, output_file="sheet_music.mid")
