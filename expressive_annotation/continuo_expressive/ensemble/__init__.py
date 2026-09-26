"""Pure functions that combine several models' outputs into one field: the age
cascade, the Chinese dialect cascade, and the emotion confidence gate. No torch here,
so they can be re-run over stored predictions without a GPU."""
