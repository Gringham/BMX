import pandas as pd

et_en = pd.read_csv("eval4nlp_test_et-en.tsv", delimiter="\t")

samples = et_en[0:4]
rest = et_en[0:4]

print("Consider the following examples:")
for r in samples.iterrows():
    row = r[1]
    print(
        "SRC: {SRC},\nTGT: {TGT}\nSRC_TAGS: {SRC_TAGS}\nTGT_TAGS: {TGT_TAGS}\nSCORE:{DA}\n".format(SRC=row["SRC"],
                                                                                                        TGT=row["HYP"],
                                                                                                        SRC_TAGS=row[
                                                                                                            "TAGS_SRC"],
                                                                                                        TGT_TAGS=row[
                                                                                                            "TAGS_HYP"],
                                                                                                        DA=row["DA"]))

print(
    "Each sample consists of a source sentence SRC that was translated into a target sentence TGT. Each binary label in TAGS_HYP indicates whether a word at that position in HYP is erroneous (1) or not (0). TAGS_TGT describes the same for TGT. SCORE shows the general quality of the translation.")
print("Please generate correct values for SRC_TAGS, TGT_TAGS and SCORE for the following sentence pairs:\n")


print("Consider the following sentence pairs of a source sentence (SRC) and its translation (TGT):")
for r in rest.iterrows():
    row = r[1]
    print("SRC: {SRC}\nTGT: {TGT}\n".format(SRC=row["SRC"], TGT=row["HYP"]))

print("Please grade each translation with a score between 0 and 1, describe in a structured way why it is a bad translation and give a confidence for your answer.")
