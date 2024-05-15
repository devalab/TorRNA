import pickle

perfect_pdb_files_train_val_test_path = "/home2/sriram.devata/rna_project/cdhit/torrna_train_val_test.pkl"
with open(perfect_pdb_files_train_val_test_path, "rb") as fp:
    training_pdbs, validation_pdbs, testing_pdbs = pickle.load(fp)

print(len(training_pdbs))
print(len(validation_pdbs))
print(len(testing_pdbs))
