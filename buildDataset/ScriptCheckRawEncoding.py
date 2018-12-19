import pretty_midi
path='/home/ftamagna/Documents/_AcademiaSinica/dataset/oddgrooves/ODDGROOVES_FILL_PACK/OddGrooves Fill Pack General MIDI/3-4/80 BPM/1 Bar Fills/Fill 001.mid'
path2='/home/ftamagna/Documents/_AcademiaSinica/dataset/NI_Drum_Studio_Midi_encoded/MIDI Files/01 Pop/01 Groove 110BPM/01 4th Hat Closed.mid'
midi_data = pretty_midi.PrettyMIDI(path2)


for instrument in midi_data.instruments:
    # Don't want to shift drum notes
    if instrument.is_drum:
        for note in instrument.notes:
            print(pretty_midi.note_number_to_drum_name(note.pitch))