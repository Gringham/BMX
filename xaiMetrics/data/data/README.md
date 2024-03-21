# Data loading

Every dataset should be represented as a TSV file with the following columns (at minimum):  
HYP - The hypothesis sentence  
SRC - The source sentence  
DA - The human score (might also be an mqm score, but should still be called DA)
REF - A human reference sentence (might also be a dummy)
SYSTEM - The machine translation system (might also be a dummy)
LP - The current language pair

We include scripts that automatically generate this kind of TSV file for the datasets we considered in our paper.
These scripts load the original dataset representations into these forms. The data has to be downloaded from the 
respective locations first. The scripts contain references to the literature the datasets originate from.