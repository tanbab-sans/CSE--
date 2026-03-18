import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os, time
import numpy as np
from scipy import sparse

from gp_torch import GaussianProcessRegressorTorch

from utils import (
    safran_process_dataset,
    get_dataset_name,
    parse_data,
    snapshotsPOD_fit_transform
)

import torch
device = torch.device("cpu")

num_steps = [1_000]
step_size = 1.

dataset_names = ["scarce_train", "full_test", "reynolds_test"]
output_names = ["velocity", "pressure", "turbulent_viscosity"]
pred_output_names = ["x-velocity", "y-velocity", "pressure", "turbulent_viscosity"]


nb_modes_shape_embedding_ = {}
nb_modes_shape_embedding_["velocity"] = 16
nb_modes_shape_embedding_["pressure"] = 16
nb_modes_shape_embedding_["turbulent_viscosity"] = 16

nb_modes_fields_ = {}
nb_modes_fields_["velocity"] = 13
nb_modes_fields_["pressure"] = 24
nb_modes_fields_["turbulent_viscosity"] = 12

kept_velocity_indices_pres = 13
kept_velocity_indices_pred = 8

common_mesh_id_ = 93


kept_indices = np.hstack((np.arange(89), np.arange(90,103))) # removing outlier


class AugmentedSimulator:
    def __init__(
        self,
        benchmark,
        nb_modes_shape_embedding = nb_modes_shape_embedding_,
        nb_modes_fields = nb_modes_fields_,
        common_mesh_id = common_mesh_id_
    ):

        self.nb_modes_shape_embedding = nb_modes_shape_embedding
        self.nb_modes_fields = nb_modes_fields
        self.common_mesh_id = common_mesh_id

        self.name = "AirfRANSSubmission"

        self.common_mesh_precomp = None

        self.sizes_meshes = {}
        self.iffo = {}

        self.X_scalars = {}
        self.scalar_scalers = None
        self.pca_clouds = None

        self.X = {}
        self.scalerX = None

        self.X2 = {}
        self.scalerX2 = None

        self.pca_fields = None
        self.y_train = None
        self.angles_train = None
        self.pca_X_angles_train = None

        self.y_scalers = None
        self.angles_scaler = None

        self.kmodels = None
        self.kmodel_angle = None

        self.angle_predictor = None

        self.path_to_simulations = benchmark.benchmark_path


    def treat_dataset(self, data, common_mesh_precomp, dataset_name, training):


        if training:
            clouds, scalars_inputs, fields, angles, X_angles = parse_data(data, training=True)
        else:
            clouds, scalars_inputs, inverse_fe_op, sizes_meshes = parse_data(data, training=False)

        if training:

            self.common_mesh_precomp = common_mesh_precomp

            # treat wake angles for learning
            self.angles_scaler = MinMaxScaler()
            self.angles_train = self.angles_scaler.fit_transform(angles.reshape(-1, 1))

            self.pca_X_angles_train, X_angles_train = snapshotsPOD_fit_transform(X_angles, sparse.eye(X_angles.shape[1]), 8)

            U = np.empty((fields.shape[0], 2*fields.shape[1]))
            for i in range(fields.shape[0]):
                U[i,:fields.shape[1]] = fields[i,:,0]
                U[i,fields.shape[1]:] = fields[i,:,1]

            self.correlationOperator1c = sparse.eye(fields.shape[1])
            self.correlationOperator2c = sparse.eye(2*fields.shape[1])

            print("Dimensionality reduction for the shape embedding:")
            self.scalar_scalers = StandardScaler()
            self.X_scalars[dataset_name] = self.scalar_scalers.fit_transform(scalars_inputs)

            self.pca_clouds = []
            X_pca = []
            for name, nbm in self.nb_modes_shape_embedding.items():
                print("shape embedding "+name)
                pca_c, x_pca = snapshotsPOD_fit_transform(clouds, self.correlationOperator2c, nbm)

                self.pca_clouds.append(pca_c)
                X_pca.append(x_pca)

            unscaled_X = [np.concatenate([xtpca, self.X_scalars[dataset_name]], axis=-1) for xtpca in X_pca]
            self.scalerX = [StandardScaler() for _ in range(len(self.nb_modes_shape_embedding))]
            self.X[dataset_name] = [scx.fit_transform(xt) for scx, xt in zip(self.scalerX, unscaled_X)]

            self.X_angle_embed_train = np.hstack((X_angles_train, self.X_scalars[dataset_name]))

            print("Dimensionality reduction for the output fields:")
            self.pca_fields = []
            self.y_scalers = []
            self.y_train = []

            print("U bulk")
            pca_f, y_pca = snapshotsPOD_fit_transform(U, self.correlationOperator2c, self.nb_modes_fields[output_names[0]])
            self.pca_fields.append(pca_f)
            y_scaler = MinMaxScaler()
            y_pca = y_scaler.fit_transform(y_pca)
            self.y_scalers.append(y_scaler)
            self.y_train.append(y_pca)

            for i in range(1, 3):
                print(output_names[i]+" bulk")
                pca_f, y_pca = snapshotsPOD_fit_transform(fields[:, :, i+1], self.correlationOperator1c, self.nb_modes_fields[output_names[i]])
                self.pca_fields.append(pca_f)
                y_scaler = MinMaxScaler()
                y_pca = y_scaler.fit_transform(y_pca)
                self.y_scalers.append(y_scaler)
                self.y_train.append(y_pca)

        else:

            self.sizes_meshes[dataset_name] = sizes_meshes
            self.iffo[dataset_name] = inverse_fe_op

            X_pca = [np.dot(self.pca_clouds[i], self.correlationOperator2c.dot(clouds.T)).T
                    for i in range(len(self.nb_modes_shape_embedding))]

            self.X_scalars[dataset_name] = self.scalar_scalers.transform(scalars_inputs)

            unscaled_X = [np.concatenate([X_pca[i], self.X_scalars[dataset_name]], axis=-1)
                    for i in range(len(self.nb_modes_shape_embedding))]

            self.X[dataset_name] = [self.scalerX[i].transform(unscaled_X[i])
                    for i in range(len(self.nb_modes_shape_embedding))]

        return None


    def process_dataset(self, dataset, training: bool) -> None:
        dataset_name = get_dataset_name(dataset, training)

        print(f"Processing dataset {dataset_name}")

        start = time.time()
        data, common_mesh_precomp = safran_process_dataset(dataset, training, self.path_to_simulations, self.common_mesh_id, self.angle_predictor, self.common_mesh_precomp)
        self.treat_dataset(data, common_mesh_precomp, dataset_name, training)
        print(f"Dataset {dataset_name} preprocessed in {int(time.time() - start)} s")



    def train(self, train_dataset, save_path=None) -> None:

        self.process_dataset(dataset=train_dataset, training=True)

        start = time.time()
        # angles
        print(f">> training angles") # using the same number of shape embedding as the velocity

        # Filter kept_indices to stay within current dataset bounds for verification/small runs
        current_nb_samples = self.X_angle_embed_train.shape[0]
        actual_kept_indices = kept_indices[kept_indices < current_nb_samples]

        X_ = self.X_angle_embed_train[actual_kept_indices,:]
        Y_ = self.angles_train[actual_kept_indices]

        input_dim = X_.shape[-1]

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        kmodel_try = GaussianProcessRegressor(
            kernel = ConstantKernel()*Matern(nu=2.5, length_scale=np.ones(input_dim), length_scale_bounds=(1e-12, 1e12)) + WhiteKernel(noise_level_bounds=(1e-12, 1e12)),
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=100,
            random_state = 0).fit(X_, Y_)

        self.kmodel_angle = kmodel_try

        self.angle_predictor = [self.pca_X_angles_train, self.scalar_scalers, self.kmodel_angle, self.angles_scaler]


        # rest
        self.kmodels = []

        i = 0
        print(f">> training {output_names[i]}")
        X_ = torch.tensor(self.X["scarce_train"][i][actual_kept_indices,:], dtype=torch.float64).to(device)

        input_dim = X_.shape[-1]
        output_dim = self.y_train[i].shape[-1]

        kmodel_i = []

        from tqdm import tqdm
        for j in tqdm(range(output_dim), desc=f"Training {output_names[i]} modes", leave=True):

            Y_ = torch.tensor(self.y_train[i][actual_kept_indices, j : j + 1], dtype=torch.float64).to(device)

            kmodel = []

            for num_step in num_steps:

                kmodel_try = GaussianProcessRegressorTorch(length_scale=np.ones(input_dim), noise_scale=1., amplitude_scale=1.).to(device)
                hist = kmodel_try.fit(X_, Y_, torch.optim.AdamW(params=kmodel_try.parameters(), lr=step_size), num_step)
                print(f"Coord. {j} | num_step : {num_step} | NLL : {hist[-1]}")
                kmodel_try.eval()

                kmodel.append(kmodel_try)

            kmodel_i.append(kmodel)

        self.kmodels.append(kmodel_i)

        self.scalerX2 = [StandardScaler() for _ in range(2)]
        unscaled_X2 = [np.hstack((self.X["scarce_train"][i][actual_kept_indices,:], self.y_train[0][actual_kept_indices,:kept_velocity_indices_pres])) for i in range(1, 3)]
        self.X2["scarce_train"] = [scx.fit_transform(xt) for scx, xt in zip(self.scalerX2, unscaled_X2)]

        for i in range(1, 3):
            print(f">> training {output_names[i]}")

            X_ = torch.tensor(self.X2["scarce_train"][i-1], dtype=torch.float64).to(device)

            input_dim = X_.shape[-1]
            output_dim = self.y_train[i].shape[-1]

            kmodel_i = []

            for j in tqdm(range(output_dim), desc=f"Training {output_names[i]} modes", leave=True):

                Y_ = torch.tensor(self.y_train[i][actual_kept_indices, j : j + 1], dtype=torch.float64).to(device)

                kmodel = []
                for num_step in num_steps:

                    kmodel_try = GaussianProcessRegressorTorch(length_scale=np.ones(input_dim), noise_scale=1., amplitude_scale=1.).to(device)
                    hist = kmodel_try.fit(X_, Y_, torch.optim.AdamW(params=kmodel_try.parameters(), lr=step_size), num_step)
                    print(f"Coord. {j} | num_step : {num_step} | NLL : {hist[-1]}")
                    kmodel_try.eval()

                    kmodel.append(kmodel_try)

                kmodel_i.append(kmodel)

            self.kmodels.append(kmodel_i)

        print(f"time to train GPs = {int(time.time() - start)} s")
        return None


    def predict(self, dataset, **kwargs):
        self.process_dataset(dataset=dataset, training=False)

        start = time.time()
        dataset_name = get_dataset_name(dataset, False)

        y_pred_common = []

        i = 0
        X_ = torch.tensor(self.X[dataset_name][i], dtype=torch.float64).to(device)

        output_dim = self.y_train[i].shape[-1]
        n_samples = self.X[dataset_name][i].shape[0]
        y_pred_i = np.empty((n_samples, output_dim, len(num_steps)))

        for j in range(output_dim):
            for k in range(len(num_steps)):
                y_pred_i[:,j,k] = self.kmodels[i][j][k](X_).detach().cpu().numpy().squeeze()

        coefs_U_pred_i = np.mean(y_pred_i, axis = 2)
        y_pred_i = self.y_scalers[i].inverse_transform(coefs_U_pred_i)

        y_pred_common_i = np.dot(y_pred_i[:,:kept_velocity_indices_pred], self.pca_fields[i][:kept_velocity_indices_pred,:])

        y_pred_common.append(y_pred_common_i[:,:int(y_pred_common_i.shape[1]/2)])
        y_pred_common.append(y_pred_common_i[:,int(y_pred_common_i.shape[1]/2):])


        unscaled_X2 = [np.hstack((self.X[dataset_name][i], coefs_U_pred_i[:,:kept_velocity_indices_pres])) for i in range(1, 3)]
        self.X2[dataset_name] = [self.scalerX2[i].transform(unscaled_X2[i]) for i in range(2)]

        for i in range(1, 3):

            X_ = torch.tensor(self.X2[dataset_name][i-1], dtype=torch.float64).to(device)

            output_dim = self.y_train[i].shape[-1]
            n_samples = self.X[dataset_name][i].shape[0]
            y_pred_i = np.empty((n_samples, output_dim, len(num_steps)))

            for j in range(output_dim):
                for k in range(len(num_steps)):
                    y_pred_i[:,j,k] = self.kmodels[i][j][k](X_).detach().cpu().numpy().squeeze()

            y_pred_i = np.mean(y_pred_i, axis = 2)
            y_pred_i = self.y_scalers[i].inverse_transform(y_pred_i)

            y_pred_common_i = np.dot(y_pred_i, self.pca_fields[i])

            y_pred_common.append(y_pred_common_i)


        inv_projected_fields = []
        for j in range(len(self.sizes_meshes[dataset_name])):
            input_fields = []
            for i in range(len(pred_output_names)):
                input_fields.append(y_pred_common[i][j])
            input_fields = np.array(input_fields)

            indices_nodes, filtered_triangles, all_coord_bary = self.iffo[dataset_name][j][0], self.iffo[dataset_name][j][1], self.iffo[dataset_name][j][2]

            inv_proj = np.empty((indices_nodes.shape[0], len(pred_output_names)))
            inv_proj[indices_nodes, :] = np.einsum('ijk,jk->ji', input_fields[:,filtered_triangles], all_coord_bary, optimize = True)

            inv_projected_fields.append(inv_proj.T)

        predictions = {}
        for i in range(len(pred_output_names)):
            y_pred_fields = []
            for j in range(len(self.sizes_meshes[dataset_name])):

                y_pred_field = inv_projected_fields[j][i]

                y_pred_filtered = y_pred_field[: self.sizes_meshes[dataset_name][j]]
                y_pred_fields.append(y_pred_filtered)

            predictions[pred_output_names[i]] = np.concatenate(y_pred_fields, axis=0)

        print("duration inference ex data processing =", time.time() - start)

        return predictions