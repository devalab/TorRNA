import os
import torch
import numpy as np
import math
from tqdm import tqdm
import wandb
import pickle

from data.rna_object import RNA
from data.dataloader import get_train_test_dataloaders, RNATorsionalAnglesDataset, collate_fn

from torch.utils.data import DataLoader

from models.transformer import TorsionalAnglesTransformerDecoder

torch.manual_seed(0)
np.random.seed(0)

pdbs_path="./data/all_torrna_pdbs/"
processed_dir = "./data/temp_dataset/"
perfect_pdb_files_train_val_test_path="./data/torrna_train_val_test.pkl"

with open(perfect_pdb_files_train_val_test_path, "rb") as fp:
        training_pdbs, validation_pdbs, testing_pdbs = pickle.load(fp)
training_files = list()
for each_training_pdb in training_pdbs:
        training_files.append(f"{pdbs_path}/{each_training_pdb}.pdb")
testing_files = list()
for each_testing_pdb in testing_pdbs:
        testing_files.append(f"{pdbs_path}/{each_testing_pdb}.pdb")

pdbs_to_predict = 1000
list_of_pdbs_to_predict = testing_files[:pdbs_to_predict]
predict_dataset = RNATorsionalAnglesDataset(list_of_pdbs_to_predict, processed_dir=processed_dir, type_dataset="test")
predict_dataloader = DataLoader(predict_dataset, collate_fn=collate_fn, batch_size=32)

train_dataset = RNATorsionalAnglesDataset(training_files, processed_dir=processed_dir, type_dataset="train")
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32)


if os.path.exists(f"./miscellaneous/rna_objects_temp.pkl"):
        with open(f"./miscellaneous/rna_objects_temp.pkl", "rb") as input_file:
                print(f"Found RNA objects precomputed")
                rna_objects = pickle.load(input_file)
else:
        rna_objects = list()
        print(f"Making RNA objects")
        for each_pdb_file in tqdm(list_of_pdbs_to_predict):
                try:
                        rna_objects.append(RNA(each_pdb_file, calc_rna_fm_embeddings=True, load_dssr_dihedrals=True, load_coords=False))
                except:
                        pass
        with open(f"./miscellaneous/rna_objects_temp.pkl", "wb") as output_file:
                pickle.dump(rna_objects, output_file)
        
if os.path.exists(f"./miscellaneous/predict_seq.fasta"):
        print(f"Found precomputed FASTA sequences")
else:
        print(f"Computing FASTA sequences for SPOT-RNA-1D")
        with open(f"./miscellaneous/predict_seq.fasta", "w") as fp:
                for ran_object_idx,each_rna_object in enumerate(rna_objects):
                        pdb_code = each_rna_object.pdb_path.split('/')[-1].replace(".pdb", "")
                        fp.write(f">{pdb_code}\n")
                        fp.write(f"{each_rna_object.dssr_full_seq}\n")
        print(f"Run SPOT-RNA-1D")
        exit(1)

# ------------------- Helper functions -------------------

all_dihedral_angle_names = ["alpha" ,"beta" ,"gamma" ,"delta" ,"chi" ,"epsilon" ,"zeta" ,"eta" ,"theta"]

def convert_to_rad_and_back_to_degree(cur_res_spot_rna_pred_angles):

        final_degree_angles = list()
        for each_angle in cur_res_spot_rna_pred_angles:
                torsion_angle = torch.tensor([each_angle * math.pi / 180])
                final_degree_angle = torch.arctan2(torch.sin(torsion_angle),torch.cos(torsion_angle)).item() * 180 / math.pi
                final_degree_angles.append(final_degree_angle)

        return final_degree_angles

def make_rad_angle_tensor_to_deg_angle_list(rad_angle_tensor):

        constructed_angles = list()
        for each_angle_idx in range(len(all_dihedral_angle_names)):
                cos_angle = rad_angle_tensor[2*each_angle_idx]
                sin_angle = rad_angle_tensor[2*each_angle_idx+1]
                rad_angle = torch.arctan2(sin_angle,cos_angle)
                deg_angle = rad_angle.item() * 180 / math.pi
                constructed_angles.append(deg_angle)

        return constructed_angles

def calculate_mae(predicted, groundtruth, prediction_method="Unknown"):

        angle_errors_to_export = list()
        all_angle_errors = list()
        for each_angle_idx in range(len(all_dihedral_angle_names)):
                all_angle_errors.append(list())
                angle_errors_to_export.append(list())

        for each_rna_idx in range(len(predicted)):

                for each_angle_idx in range(len(all_dihedral_angle_names)):
                        angle_errors_to_export[each_angle_idx].append(list())

                for each_residue_idx in range(len(predicted[each_rna_idx])):

                        predicted_angles = predicted[each_rna_idx][each_residue_idx]
                        gt_angles = groundtruth[each_rna_idx][each_residue_idx]

                        for each_angle_idx in range(len(all_dihedral_angle_names)):

                                if prediction_method == "Random Baseline":
                                        difference = abs(predicted_angles[each_angle_idx] - gt_angles[each_angle_idx])
                                        difference = sum(difference)/len(difference)
                                        difference = difference.item()
                                else:
                                        difference = abs(predicted_angles[each_angle_idx] - gt_angles[each_angle_idx])

                                difference = min(difference, 360-difference)
                                if math.isnan(difference):
                                        continue
                                all_angle_errors[each_angle_idx].append(difference)
                                angle_errors_to_export[each_angle_idx][-1].append(difference)

        return_maes = list()
        all_errors = list()

        print(f"\n----------\nMAEs for all dihedral angles predicted by {prediction_method}")
        for each_angle_idx in range(len(all_dihedral_angle_names)):
                mae = sum(all_angle_errors[each_angle_idx])/len(all_angle_errors[each_angle_idx])
                return_maes.append(mae)
                all_errors.append(all_angle_errors[each_angle_idx])
                print(f"{all_dihedral_angle_names[each_angle_idx]}: {mae:.3f}")

        return return_maes, all_errors, angle_errors_to_export


# ------------------- Predict using transformer -------------------

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--wandb', default="disabled")
parser.add_argument('-ti', '--trainingiters', type=int, default=100)
parser.add_argument('-lr', '--lr', type=float, default=2e-4)
parser.add_argument('-ed', '--embeddim', type=int, default=640)
parser.add_argument('-hd', '--hiddendim', type=int, default=256)
parser.add_argument('-nh', '--numheads', type=int, default=4)
parser.add_argument('-nl', '--numlayers', type=int, default=3)
parser.add_argument('-do', '--dropout', type=float, default=0.2)
parser.add_argument('-tol', '--earlystoppingtolerance', type=int, default=5)
args = parser.parse_args()


model = TorsionalAnglesTransformerDecoder(embed_dim=args.embeddim, hidden_dim=args.hiddendim, num_heads=args.numheads, num_layers=args.numlayers, dropout=args.dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(f"./checkpoints/torrna_best_model_rna_transformer_{args.lr}_{args.embeddim}_{args.hiddendim}_{args.numheads}_{args.numlayers}_{args.dropout}_{args.earlystoppingtolerance}.pkl", map_location=device))
model.eval()
print(model)
print(f"./checkpoints/best_model_rna_transformer_{args.lr}_{args.embeddim}_{args.hiddendim}_{args.numheads}_{args.numlayers}_{args.dropout}_{args.earlystoppingtolerance}.pkl")

all_predicted_angles = list()
all_gt_angles = list()
for each_batch in predict_dataloader:
                rna_fm_embeddings, torsional_angles, initial_embeddings, padding_mask = each_batch
                rna_fm_embeddings = rna_fm_embeddings.to(device)
                torsional_angles = torsional_angles.to(device)
                initial_embeddings = initial_embeddings.to(device)
                padding_mask = padding_mask.to(device)

                output = model(rna_fm_embeddings, padding_mask, initial_embeddings).detach().cpu()
                torsional_angles = torsional_angles.detach().cpu()

                for each_rna_idx in range(len(torsional_angles)):
                        each_rna_predicted_angles = list()
                        each_rna_gt_angles = list()
                        start_range = 1
                        if torch.sum(~padding_mask[each_rna_idx, 1:]).cpu().item() == len(torsional_angles[each_rna_idx]):
                                start_range = 0
                        for each_residue_idx in range(start_range, len(torsional_angles[each_rna_idx])): # ignore angles of the first residue, start_range is to handle the case when it is the longest sequence in the batch

                                if padding_mask[each_rna_idx][each_residue_idx]:
                                        continue        # True if this residue is padded
                    
                                predicted_angles = make_rad_angle_tensor_to_deg_angle_list(output[each_rna_idx][each_residue_idx])
                                gt_angles = make_rad_angle_tensor_to_deg_angle_list(torsional_angles[each_rna_idx][each_residue_idx])
                                each_rna_predicted_angles.append(predicted_angles)
                                each_rna_gt_angles.append(gt_angles)

#                       print(f"Transformer ernapred: {len(each_rna_predicted_angles)}, ernagt: {len(each_rna_gt_angles)}")
                        all_predicted_angles.append(each_rna_predicted_angles)
                        all_gt_angles.append(each_rna_gt_angles)

all_maes, all_model_errors, angle_errors_to_export = calculate_mae(all_predicted_angles, all_gt_angles, prediction_method="RNA Transformer Decoder")
with open(f"./miscellaneous/model_mae_errors_compare_predictions.pkl", "wb") as output_file:
        pickle.dump((all_maes, all_model_errors, angle_errors_to_export), output_file)

# ------------------- Check SPOT-RNA-1D predictions -------------------

all_predicted_angles = list()
all_gt_angles = list()

spot_rna_pred_location = "./miscellaneous/SPOT-RNA-1D/outputs/"
for each_pdb_idx in range(len(list_of_pdbs_to_predict)):

        try:
                pdb_file_name = list_of_pdbs_to_predict[each_pdb_idx]
                rna_object = rna_objects[each_pdb_idx]
                gt_angles = rna_object.dssr_torsion_angles[1:]
                # print(pdb_file_name, len(rna_object.dssr_full_seq))

                pdb_code = pdb_file_name.split('/')[-1].replace(".pdb", "")
                spot_rna_pred_file = f"{spot_rna_pred_location}/{pdb_code}.txt"

                with open(spot_rna_pred_file, "r") as f:
                        spot_rna_pred_lines = f.readlines()

                each_rna_predicted_angles = list()
                each_rna_gt_angles = list()
                # print(pdb_code, gt_angles.shape)
                for each_residue_idx in range(len(gt_angles)):
                        spot_rna_pred_line = spot_rna_pred_lines[each_residue_idx+2]    # first two lines are column headers and an empty line
                        cur_res_spot_rna_pred_angles = [float(x) for x in spot_rna_pred_line.strip().split()[2:]]       # first two columns are number and basename

                        # cur_res_spot_rna_pred_angles = convert_to_rad_and_back_to_degree(cur_res_spot_rna_pred_angles)
                        
                        # correcting the order of the angles, SPOT-RNA-1D has a different ordering of angles
                        index_correction = [0, 1, 2, 3, 6, 4, 5, 7, 8]
                        cur_res_spot_rna_pred_angles = [cur_res_spot_rna_pred_angles[x] for x in index_correction]

                        cur_res_gt_angles = make_rad_angle_tensor_to_deg_angle_list(gt_angles[each_residue_idx])

                        each_rna_predicted_angles.append(cur_res_spot_rna_pred_angles)
                        each_rna_gt_angles.append(cur_res_gt_angles)

                        # print("Pred:", cur_res_spot_rna_pred_angles)
                        # print("GT  :", cur_res_gt_angles)
#               print(f"SPOT ernapred: {len(each_rna_predicted_angles)}, ernagt: {len(each_rna_gt_angles)}")
                all_predicted_angles.append(each_rna_predicted_angles)
                all_gt_angles.append(each_rna_gt_angles)
        except Exception as e:
                pass

all_maes, all_spot_errors, angle_errors_to_export = calculate_mae(all_predicted_angles, all_gt_angles, prediction_method="SPOT-RNA-1D")
with open(f"./miscellaneous/spot_rna_mae_errors_compare_predictions.pkl", "wb") as output_file:
        pickle.dump((all_maes, all_spot_errors, angle_errors_to_export), output_file)

# -------------- Random Baseline Prediction -----------
each_dihedral_all_angles = list()
for each_angle_idx in range(len(all_dihedral_angle_names)):
        each_dihedral_all_angles.append(list())
    
    
all_train_gt_angles = list()
for each_batch in train_dataloader:
                rna_fm_embeddings, torsional_angles, initial_embeddings, padding_mask = each_batch
                torsional_angles = torsional_angles.to(device)

                for each_rna_idx in range(len(torsional_angles)):
                        each_rna_gt_angles = list()
                        start_range = 1
                        if torch.sum(~padding_mask[each_rna_idx, 1:]).cpu().item() == len(torsional_angles[each_rna_idx]):
                                start_range = 0
                        for each_residue_idx in range(start_range, len(torsional_angles[each_rna_idx])): # ignore angles of the first residue, start_range is to handle the case when it is the longest sequence in the batch

                                gt_angles = make_rad_angle_tensor_to_deg_angle_list(torsional_angles[each_rna_idx][each_residue_idx])
                                each_rna_gt_angles.append(gt_angles)

                        all_train_gt_angles.append(each_rna_gt_angles)

for each_rna_idx in range(len(all_train_gt_angles)):
        for each_residue_idx in range(len(all_train_gt_angles[each_rna_idx])):
                gt_angles = all_train_gt_angles[each_rna_idx][each_residue_idx]
                for each_angle_idx in range(len(all_dihedral_angle_names)):
                        each_dihedral_all_angles[each_angle_idx].append(gt_angles[each_angle_idx])

each_dihedral_hist = list()
for each_angle_idx in range(len(all_dihedral_angle_names)):
        hist_freqs,buckets = np.histogram([x for x in each_dihedral_all_angles[each_angle_idx] if not math.isnan(x) and x != 0], bins=180, density=True)
        each_dihedral_hist.append((hist_freqs,buckets))


all_random_baseline_predicted = list()
for each_rna_idx in range(len(all_gt_angles)):
        each_rna_random_baseline_preds = list()
        for each_residue_idx in range(len(all_gt_angles[each_rna_idx])):

                each_residue_random_baseline_preds = list()
                gt_angles = all_gt_angles[each_rna_idx][each_residue_idx]

                for each_angle_idx in range(len(all_dihedral_angle_names)):
                        hist_freqs,buckets = each_dihedral_hist[each_angle_idx]
                        random_baseline_predictions = np.random.choice((buckets[:-1] + buckets[1:])/2, size=100, p=hist_freqs/hist_freqs.sum())
                        # predicted_angle = np.mean(random_baseline_predictions)
                        predicted_angle = torch.tensor(random_baseline_predictions)
                        each_residue_random_baseline_preds.append(predicted_angle)

                each_rna_random_baseline_preds.append(each_residue_random_baseline_preds)
        all_random_baseline_predicted.append(each_rna_random_baseline_preds)
    
all_maes, all_random_errors, angle_errors_to_export = calculate_mae(all_random_baseline_predicted, all_gt_angles, prediction_method="Random Baseline")
with open(f"./miscellaneous/random_baseline_mae_errors_compare_predictions.pkl", "wb") as output_file:
        pickle.dump((all_maes, all_random_errors, angle_errors_to_export), output_file)

# ------------------ Calculating t-test ------------
from scipy.stats import ttest_rel

print(f"\n----------\nTTest Results (P-value) for for all dihedral angles predicted by the model and by SPOT-RNA-1D")
for each_angle_idx in range(len(all_model_errors)):
        # print(len(all_model_errors[each_angle_idx]))
        # print(len(all_spot_errors[each_angle_idx]))
        ttest_result = ttest_rel(all_model_errors[each_angle_idx], all_spot_errors[each_angle_idx], alternative='less')
        print(f"{all_dihedral_angle_names[each_angle_idx]}: {ttest_result.pvalue}")
        
        
        
