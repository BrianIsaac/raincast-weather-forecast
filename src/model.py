import gzip
import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.models import TimeXer
from lightning.pytorch import Trainer as LTrainer

# sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

########## SARIMA Implementation ##########

def fit_and_save_sarimax(
    train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    exog: pd.DataFrame = None,
    model_dir: str = "models/SARIMA"
):
    """
    Fit a SARIMAX model, then save:
      - model_full.pth       : the full SARIMAXResults (remove_data=False)
      - model.pth.gz         : the stripped, compressed version (remove_data=True + gzip)

    Returns:
        results       : the SARIMAXResults object
        full_path     : path to the full .pth
        compressed_gz : path to the stripped & compressed .pth.gz
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1) Fit
    model   = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # 2) Save the full model (large)
    full_path = os.path.join(model_dir, "model_full.pth")
    results.save(full_path)  # remove_data defaults to False

    # 3) Save the stripped model
    stripped_path = os.path.join(model_dir, "model.pth")
    results.save(stripped_path, remove_data=True)

    # 4) Compress the stripped model
    compressed_gz = stripped_path + ".gz"
    with gzip.open(compressed_gz, "wb") as f_out:
        with open(stripped_path, "rb") as f_in:
            f_out.write(f_in.read())

    return results, full_path, compressed_gz

########## TLSTM PyTorch Implementation ##########

# ──────────────────────────────────────────────────────────────────────────────
# 1) TLSTM cell (shared by both models)
# ──────────────────────────────────────────────────────────────────────────────
class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wi,  self.Ui,  self.bi   = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Parameter(torch.ones(hidden_dim))
        self.Wf,  self.Uf,  self.bf   = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Parameter(torch.ones(hidden_dim))
        self.Wog, self.Uog, self.bog  = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Parameter(torch.ones(hidden_dim))
        self.Wc,  self.Uc,  self.bc   = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Parameter(torch.ones(hidden_dim))
        self.Wd,  self.bd       = nn.Linear(hidden_dim, hidden_dim), nn.Parameter(torch.ones(hidden_dim))

    def map_elapse_time(self, t):
        return (1.0 / torch.log(t + 2.7183)).expand(-1, self.hidden_dim)

    def forward(self, x_seq, t_seq):
        if t_seq.dim() == 3:  
            t_seq = t_seq.squeeze(-1)
        B, L, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        c = torch.zeros(B, self.hidden_dim, device=x_seq.device)

        for i in range(L):
            x_t = x_seq[:, i]
            dt  = t_seq[:, i].unsqueeze(1)
            T   = self.map_elapse_time(dt)
            CST = torch.tanh(self.Wd(c) + self.bd)
            c   = c - CST + T * CST

            i_gate = torch.sigmoid(self.Wi(x_t)  + self.Ui(h)  + self.bi)
            f_gate = torch.sigmoid(self.Wf(x_t)  + self.Uf(h)  + self.bf)
            o_gate = torch.sigmoid(self.Wog(x_t) + self.Uog(h) + self.bog)
            c_hat  = torch.tanh(self.Wc(x_t)    + self.Uc(h)  + self.bc)

            c = f_gate * c + i_gate * c_hat
            h = o_gate * torch.tanh(c)

        return h  # [B,hidden_dim]


# ──────────────────────────────────────────────────────────────────────────────
# 2) Regressor: TLSTM → 2‐layer MLP → single‐scalar output
# ──────────────────────────────────────────────────────────────────────────────
class TLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim):
        super().__init__()
        self.cell = TLSTMCell(input_dim, hidden_dim)
        self.fc1  = nn.Linear(hidden_dim, fc_dim)
        self.ln1  = nn.LayerNorm(fc_dim)
        self.fc2  = nn.Linear(fc_dim, fc_dim)
        self.ln2  = nn.LayerNorm(fc_dim)
        self.out  = nn.Linear(fc_dim, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, t):
        h = self.cell(x, t)
        y = F.gelu(self.ln1(self.fc1(h)))
        y = self.drop(y)
        y = F.gelu(self.ln2(self.fc2(y)))
        y = self.drop(y)
        return self.out(y)  # [B,1]


class TLSTMTrainer_Regressor:
    def __init__(self, input_dim, hidden_dim, fc_dim,
                 lr=1e-3, use_compile=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = TLSTMRegressor(input_dim, hidden_dim, fc_dim).to(self.device)
        if use_compile:
            self.model = torch.compile(self.model)
        self.opt    = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.lossfn = nn.MSELoss()

    def train_with_loader(self, loader, val_loader=None, epochs=100):
        for ep in range(1, epochs+1):
            self.model.train()
            tot_loss, tot_n = 0.0, 0
            for X, T, y_amt, y_occ in loader:
                X, T    = X.to(self.device), T.to(self.device)
                y_amt   = y_amt.to(self.device)          # [B,1]
                self.opt.zero_grad()
                preds   = self.model(X, T)               # [B,1]
                loss    = self.lossfn(preds, y_amt)
                loss.backward()
                self.opt.step()
                bsz     = y_amt.size(0)
                tot_loss += loss.item() * bsz
                tot_n    += bsz
            train_mse = tot_loss / tot_n

            if val_loader:
                self.model.eval()
                v_loss, v_n = 0.0, 0
                with torch.no_grad():
                    for X, T, y_amt, y_occ in val_loader:
                        X, T  = X.to(self.device), T.to(self.device)
                        y_amt = y_amt.to(self.device)
                        pred  = self.model(X, T)             # [B,1]
                        v_loss += self.lossfn(pred, y_amt).item() * y_amt.size(0)
                        v_n    += y_amt.size(0)
                val_mse = v_loss / v_n
                print(f"[Reg] Ep{ep}/{epochs}  TrMSE:{train_mse:.4f}  ValMSE:{val_mse:.4f}")
            else:
                print(f"[Reg] Ep{ep}/{epochs}  TrMSE:{train_mse:.4f}")

    def predict_loader(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, T, y_amt, y_occ in loader:
                X, T = X.to(self.device), T.to(self.device)
                p    = self.model(X, T).cpu().numpy()  # [B,1]
                preds.append(p)
                trues.append(y_amt.numpy())           # [B,1]
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
    
    def train_one_epoch(self, loader, val_loader=None):
        # single‐epoch training
        self.model.train()
        tot_loss, tot_n = 0.0, 0
        for X, T, y_amt, *_ in loader:
            X, T, y_amt = X.to(self.device), T.to(self.device), y_amt.to(self.device)
            self.opt.zero_grad()
            preds = self.model(X, T)
            loss  = self.lossfn(preds, y_amt)
            loss.backward()
            self.opt.step()
            bsz = y_amt.size(0)
            tot_loss += loss.item() * bsz
            tot_n    += bsz
        train_mse = tot_loss / tot_n
        self.train_loss = train_mse

        # single‐epoch validation (if provided)
        if val_loader is not None:
            self.model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for X, T, y_amt, *_ in val_loader:
                    X, T, y_amt = X.to(self.device), T.to(self.device), y_amt.to(self.device)
                    v_loss += self.lossfn(self.model(X, T), y_amt).item() * y_amt.size(0)
                    v_n    += y_amt.size(0)
            self.val_loss = v_loss / v_n
        else:
            self.val_loss = None

        return self.train_loss, self.val_loss

# ──────────────────────────────────────────────────────────────────────────────
# 3) Classifier: TLSTM → same body → single‐logit output
# ──────────────────────────────────────────────────────────────────────────────
class TLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim):
        super().__init__()
        self.cell = TLSTMCell(input_dim, hidden_dim)
        self.fc1  = nn.Linear(hidden_dim, fc_dim)
        self.ln1  = nn.LayerNorm(fc_dim)
        self.out  = nn.Linear(fc_dim, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, t):
        h = self.cell(x, t)
        y = F.gelu(self.ln1(self.fc1(h)))
        y = self.drop(y)
        return self.out(y)  # [B,1] logit


class TLSTMTrainer_Classifier:
    def __init__(self, input_dim, hidden_dim, fc_dim,
                 lr=1e-3, use_compile=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = TLSTMClassifier(input_dim, hidden_dim, fc_dim).to(self.device)
        if use_compile:
            self.model = torch.compile(self.model)
        self.opt    = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.lossfn = nn.BCEWithLogitsLoss()

    def train_with_loader(self, loader, val_loader=None, epochs=100):
        for ep in range(1, epochs+1):
            self.model.train()
            tot_loss, tot_n, corr = 0.0, 0, 0
            for X, T, y_amt, y_occ in loader:
                X, T   = X.to(self.device), T.to(self.device)
                y_occ  = y_occ.to(self.device)               # [B,1]
                self.opt.zero_grad()
                logit  = self.model(X, T)                    # [B,1]
                loss   = self.lossfn(logit, y_occ)
                loss.backward()
                self.opt.step()
                bsz    = y_occ.size(0)
                tot_loss += loss.item() * bsz
                tot_n    += bsz
                preds    = (torch.sigmoid(logit) > 0.5).float()
                corr    += (preds == y_occ).sum().item()

            train_bce = tot_loss / tot_n
            train_acc = corr / tot_n

            if val_loader:
                self.model.eval()
                v_loss, v_n, v_corr = 0.0, 0, 0
                with torch.no_grad():
                    for X, T, y_amt, y_occ in val_loader:
                        X, T   = X.to(self.device), T.to(self.device)
                        y_occ  = y_occ.to(self.device)
                        logit  = self.model(X, T)
                        lo     = self.lossfn(logit, y_occ)
                        bsz    = y_occ.size(0)
                        v_loss += lo.item() * bsz
                        v_n    += bsz
                        v_corr += ((torch.sigmoid(logit)>0.5).float() == y_occ).sum().item()
                val_bce = v_loss / v_n
                val_acc = v_corr / v_n
                print(f"[Cls] Ep{ep}/{epochs}  TrBCE:{train_bce:.4f}, TrAcc:{train_acc:.2%}  ValBCE:{val_bce:.4f}, ValAcc:{val_acc:.2%}")
            else:
                print(f"[Cls] Ep{ep}/{epochs}  TrBCE:{train_bce:.4f}, TrAcc:{train_acc:.2%}")

    def predict_loader(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, T, y_amt, y_occ in loader:
                X, T   = X.to(self.device), T.to(self.device)
                logit  = self.model(X, T)                 # [B,1]
                prob   = torch.sigmoid(logit).cpu().numpy()# [B,1]
                preds.append((prob > 0.5).astype(float))   # [B,1]
                trues.append(y_occ.numpy())               # [B,1]
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
    
    def train_one_epoch(self, loader, val_loader=None):
        self.model.train()
        tot_loss, tot_n, corr = 0.0, 0, 0
        for X, T, *_, y_occ in loader:
            X, T, y_occ = X.to(self.device), T.to(self.device), y_occ.to(self.device)
            self.opt.zero_grad()
            logit = self.model(X, T)
            loss  = self.lossfn(logit, y_occ)
            loss.backward()
            self.opt.step()
            bsz       = y_occ.size(0)
            tot_loss += loss.item() * bsz
            tot_n    += bsz
            corr     += ((torch.sigmoid(logit) > 0.5).float() == y_occ).sum().item()
        train_bce = tot_loss / tot_n
        train_acc = corr / tot_n
        self.train_loss = train_bce
        self.train_acc  = train_acc

        if val_loader is not None:
            self.model.eval()
            v_loss, v_n, v_corr = 0.0, 0, 0
            with torch.no_grad():
                for X, T, *_, y_occ in val_loader:
                    X, T, y_occ = X.to(self.device), T.to(self.device), y_occ.to(self.device)
                    logit = self.model(X, T)
                    v_loss += self.lossfn(logit, y_occ).item() * y_occ.size(0)
                    v_n    += y_occ.size(0)
                    v_corr += ((torch.sigmoid(logit) > 0.5).float() == y_occ).sum().item()
            self.val_loss  = v_loss / v_n
            self.val_acc   = v_corr / v_n
        else:
            self.val_loss = None
            self.val_acc  = None

        return self.train_acc, self.train_acc  # or (loss, acc), depending on your loop

# ──────────────────────────────────────────────────────────────────────────────
# 4) Ensemble wrapper: train two separate models and gate at inference
# ──────────────────────────────────────────────────────────────────────────────
class TLSTMEnsembleTrainer:
    """
    Wraps two separate TLSTM trainers—a regressor and a classifier—
    training each independently and then gating their outputs at inference.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        fc_dim: int = 32,
        lr_reg: float = 1e-3,
        lr_cls: float = 1e-3,
        use_compile: bool = True
    ):
        self.reg_trainer = TLSTMTrainer_Regressor(
            input_dim   = input_dim,
            hidden_dim  = hidden_dim,
            fc_dim      = fc_dim,
            lr          = lr_reg,
            use_compile = use_compile
        )
        self.cls_trainer = TLSTMTrainer_Classifier(
            input_dim   = input_dim,
            hidden_dim  = hidden_dim,
            fc_dim      = fc_dim,
            lr          = lr_cls,
            use_compile = use_compile
        )
        # to mirror single‐model .val_losses/.val_accuracies
        self.val_losses = []
        self.val_accuracies = []

    def train(self, train_loader, val_loader=None, epochs: int = 100):
        for _ in range(epochs):
            self.train_one_epoch(train_loader, val_loader=val_loader)

    def train_one_epoch(self, train_loader, val_loader=None):
        """
        Train both sub‐models for exactly one epoch.
        Returns:
            train_loss (float): regressor’s training loss
            train_acc  (float): classifier’s training accuracy
        """
        # 1) run one epoch on the regressor => returns (train_loss, val_loss)
        reg_train_loss, reg_val_loss = self.reg_trainer.train_one_epoch(
            train_loader, val_loader=val_loader
        )
        # 2) run one epoch on the classifier => returns (train_acc, val_acc)
        cls_train_acc, cls_val_acc = self.cls_trainer.train_one_epoch(
            train_loader, val_loader=val_loader
        )

        # record the *latest* validation metrics
        self.val_losses.append(reg_val_loss)
        self.val_accuracies.append(cls_val_acc)

        # return exactly the two metrics your loop expects
        return reg_train_loss, cls_train_acc

    def predict(self, loader):
        pred_amt, true_amt = self.reg_trainer.predict_loader(loader)
        pred_occ, _        = self.cls_trainer.predict_loader(loader)
        pred_amt = np.maximum(pred_amt, 0)
        final_pred         = pred_amt * pred_occ
        return final_pred, true_amt, pred_amt, pred_occ

    def plot(self, loader, window=None, save_path=None):
        """
        Run inference via self.predict, plot true vs. gated predictions,
        compute MAE/RMSE, and optionally save the figure.

        Parameters:
        - loader: DataLoader for inference
        - window: tuple (start, end) to slice the series
        - save_path: str or Path where to save the plot (e.g. "outputs/forecast.png")
        """
        # 1. Get predictions & truths
        final_pred, true_amt, _, _ = self.predict(loader)
        preds = final_pred.flatten()
        trues = true_amt.flatten()
        if window:
            s, e = window
            preds, trues = preds[s:e], trues[s:e]
        errs = np.abs(preds - trues)

        # 2. Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(trues, label='True', linewidth=2)
        ax.plot(preds, '--', label='Predicted (gated)')
        ax.fill_between(np.arange(len(trues)), trues, preds, alpha=0.2)
        ax.set_title("Gated Rainfall Forecast")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Rainfall (mm)")
        ax.legend()
        fig.tight_layout()

        # 3. Save if requested
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path)
            print(f"Plot saved to: {save_path}")

        # 4. Finally show & print metrics
        if not save_path:
            plt.show()
            print(f"MAE: {errs.mean():.4f}, RMSE: {np.sqrt((errs**2).mean()):.4f}")

    def plotly(self, loader, window=None, save_path=None):
        """
        Build an interactive Plotly forecast and optionally save it as an HTML file.

        Parameters:
        - loader: DataLoader for inference
        - window: tuple (start, end) to slice the series
        - save_path: str path (e.g. "outputs/forecast.html") to save the HTML

        Returns:
        - dict: Evaluation metrics {"rmse": float, "mae": float}
        """
        # 1. Inference
        final_pred, true_amt, _, _ = self.predict(loader)
        preds = final_pred.flatten()
        trues = true_amt.flatten()

        # 2. Slice if window is given
        if window:
            s, e = window
            preds, trues = preds[s:e], trues[s:e]

        # 3. Compute metrics
        rmse = root_mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        metrics = {"rmse": rmse, "mae": mae}

        # 4. Plot
        x = np.arange(len(trues))
        fig = go.Figure([
            go.Scatter(x=x, y=trues, mode='lines', name='True'),
            go.Scatter(x=x, y=preds, mode='lines', name='Predicted (gated)'),
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([trues, preds[::-1]]),
                fill='toself', fillcolor='rgba(128,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', showlegend=False
            )
        ])
        fig.update_layout(
            title="Gated Rainfall Forecast",
            xaxis_title="Sample Index",
            yaxis_title="Rainfall (mm)",
            template="plotly_white"
        )

        # 5. Save HTML if requested
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.write_html(save_path, include_plotlyjs='cdn')
            print(f"Interactive plot saved to: {save_path}")
        else:
            fig.show()

        # 6. Return evaluation metrics
        return metrics

# -------------------------------
# 1) TLSTM cell + single‐scalar head
# -------------------------------

class TLSTM_PyTorch(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Input Gate
        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Ui = nn.Linear(hidden_dim, hidden_dim)
        self.bi = nn.Parameter(torch.ones(hidden_dim))

        # Forget Gate
        self.Wf = nn.Linear(input_dim, hidden_dim)
        self.Uf = nn.Linear(hidden_dim, hidden_dim)
        self.bf = nn.Parameter(torch.ones(hidden_dim))

        # Output Gate
        self.Wog = nn.Linear(input_dim, hidden_dim)
        self.Uog = nn.Linear(hidden_dim, hidden_dim)
        self.bog = nn.Parameter(torch.ones(hidden_dim))

        # Candidate Cell
        self.Wc = nn.Linear(input_dim, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, hidden_dim)
        self.bc = nn.Parameter(torch.ones(hidden_dim))

        # Time decay decomposition
        self.W_decomp = nn.Linear(hidden_dim, hidden_dim)
        self.b_decomp = nn.Parameter(torch.ones(hidden_dim))

        # Fully connected layers
        self.fc1    = nn.Linear(hidden_dim, fc_dim)
        self.fc_out = nn.Linear(fc_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def map_elapse_time(self, t):
        # causal time‐decay factor
        T = 1.0 / torch.log(t + 2.7183)
        return T.expand(-1, self.hidden_dim)

    def forward(self, x, elapsed_time):
        """
        x: [batch, seq_len, input_dim]
        elapsed_time: [batch, seq_len] or [batch, seq_len, 1]
        """
        if elapsed_time.dim() == 3 and elapsed_time.size(-1) == 1:
            elapsed_time = elapsed_time.squeeze(-1)

        batch_size, seq_len, _ = x.size()
        device = x.device

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t_step in range(seq_len):
            x_t = x[:, t_step, :]
            delta_t = elapsed_time[:, t_step].unsqueeze(1)

            T      = self.map_elapse_time(delta_t)
            C_ST   = torch.tanh(self.W_decomp(c_t) + self.b_decomp)
            C_STd  = T * C_ST
            c_t    = c_t - C_ST + C_STd

            i = torch.sigmoid(self.Wi(x_t)   + self.Ui(h_t)   + self.bi)
            f = torch.sigmoid(self.Wf(x_t)   + self.Uf(h_t)   + self.bf)
            o = torch.sigmoid(self.Wog(x_t)  + self.Uog(h_t)  + self.bog)
            C_hat = torch.tanh(self.Wc(x_t)  + self.Uc(h_t)   + self.bc)

            c_t = f * c_t + i * C_hat
            h_t = o * torch.tanh(c_t)

        out = F.relu(self.fc1(h_t))
        out = self.dropout(out)
        out = self.fc_out(out)
        return out


# -------------------------------
# 2) Trainer with MSE loss + MATPLOTLIB & PLOTLY plotting
# -------------------------------

class TLSTMTrainer_PyTorch:
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 fc_dim,
                 learning_rate: float = 0.001,
                 dropout_prob: float = 0.2,
                 use_compile: bool = True):
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        model = TLSTM_PyTorch(input_dim, output_dim, hidden_dim, fc_dim).to(self.device)
        if use_compile:
            model = torch.compile(model)
        self.model = model

        self.criterion    = nn.MSELoss()
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.dropout_prob = dropout_prob

    def train_with_loader(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0.0
            for X, T, Y in train_loader:
                X, T, Y = X.to(self.device), T.to(self.device), Y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X, T)
                loss = self.criterion(outputs, Y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * Y.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X, T, Y in val_loader:
                        X, T, Y = X.to(self.device), T.to(self.device), Y.to(self.device)
                        outputs = self.model(X, T)
                        val_loss += self.criterion(outputs, Y).item() * Y.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f} — Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f}")

    def evaluate_with_loader(self, data_loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, T, Y in data_loader:
                X, T = X.to(self.device), T.to(self.device)
                output = self.model(X, T)
                preds.append(output.cpu().numpy())
                trues.append(Y.cpu().numpy())

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        rmse = root_mean_squared_error(trues, preds)
        print(f"Test RMSE: {rmse:.4f}")
        return preds, trues

    def plot_predictions(self, predictions, targets, window=None):
        preds = predictions.flatten()
        trues = targets.flatten()
        if window:
            start, end = window
            preds = preds[start:end]
            trues = trues[start:end]

        errs = np.abs(preds - trues)
        plt.figure(figsize=(14, 6))
        plt.plot(trues, label='True', linewidth=2)
        plt.plot(preds, label='Pred', linestyle='--')
        plt.fill_between(np.arange(len(trues)), trues, preds, alpha=0.2)
        plt.title("Pred vs True")
        plt.legend()
        plt.show()
        print(f"MAE: {errs.mean():.4f}, RMSE: {np.sqrt((errs**2).mean()):.4f}")

    def plotly_predictions(self, predictions, targets, window=None):
        preds = predictions.flatten()
        trues = targets.flatten()
        if window:
            start, end = window
            preds = preds[start:end]
            trues = trues[start:end]

        x = np.arange(len(trues))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=trues, mode='lines', name='True'))
        fig.add_trace(go.Scatter(x=x, y=preds, mode='lines', name='Pred'))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([trues, preds[::-1]]),
            fill='toself', fillcolor='rgba(128,128,128,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip', showlegend=False
        ))
        fig.update_layout(
            title="Pred vs True",
            xaxis_title="Sample Index",
            yaxis_title="Value",
            template="plotly_white"
        )
        fig.show()

########## TimeXer ##########

class TimeXerTrainer:
    """
    Encapsulates training, prediction, and plotting for a TimeXer model.
    """
    def __init__(
        self,
        training_dataset,
        train_loader,
        val_loader,
        test_df,
        window_size,
        target_col,
        trainer_params=None,
        model_params=None,
        epochs:int = 50
    ):
        """
        Args:
            training_dataset: TimeSeriesDataSet used for training.
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation/testing.
            test_df: original test DataFrame (datetime-indexed).
            window_size: look-back window size (int).
            target_col: name of the target column (str).
            trainer_params: dict of Trainer kwargs (max_epochs, gradient_clip_val, etc.).
            model_params: dict of TimeXer.from_dataset kwargs.
            epochs: number of epochs to train
        """
        self.training_dataset = training_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_df = test_df
        self.window_size = window_size
        self.target_col = target_col
        self.epochs = epochs

        # default Lightning Trainer parameters
        default_trainer = {"max_epochs": self.epochs, "gradient_clip_val": 0.1}
        trainer_params = trainer_params or default_trainer
        self.trainer = LTrainer(**trainer_params)

        # default TimeXer model parameters
        default_model = {
            "learning_rate": 1e-3,
            "hidden_size": 128,
            "n_heads": 4,
            "e_layers": 2,
            "dropout": 0.1,
            "features": "MS"
        }
        model_params = model_params or default_model
        self.model = TimeXer.from_dataset(self.training_dataset, **model_params)

    def fit(self):
        """Train the model on the training loader and validate on the validation loader."""
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def predict(self):
        """Generate t+1 predictions for all windows in the validation loader."""
        preds = self.model.predict(self.val_loader).squeeze()
        # move to CPU numpy if needed
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        return preds

    def plot(self, save_path: str = None):
        """
        Plot predicted vs. actual values across the test period, and optionally
        save the figure to disk.

        Args:
            save_path (str, optional): where to write the .png (e.g. "outputs/forecast.png").
                                       If None, no file is written.
        """
        # 1) Generate preds & truths
        preds = self.predict()
        truths = self.test_df[self.target_col].to_numpy()[
            self.window_size : self.window_size + len(preds)
        ]
        times = self.test_df.index[
            self.window_size : self.window_size + len(preds)
        ]

        # 2) Build plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, truths, label="actual rainfall")
        ax.plot(times, preds,   label="TimeXer forecast", alpha=0.7)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(f"{self.target_col} (t+1)")
        ax.legend()
        fig.tight_layout()

        # 3) Save if requested
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path)
            print(f"Plot saved to: {save_path}")

        # 4) Show
        if not save_path:    
            plt.show()

    def run(self):
        """
        Convenience method: fit the model, then plot results.
        Returns the raw predictions array.
        """
        self.fit()
        self.plot()
        return self.predict()
    
########## Random Forest ##########

class RandomForestTrainer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "target",
        **rf_kwargs
    ):
        self.target_col = target_col

        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]

        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]

        self.model = RandomForestRegressor(**rf_kwargs)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> dict:
        preds = self.model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, preds)
        metrics = {
            "r2": r2_score(self.y_test, preds),
            "mae": mean_absolute_error(self.y_test, preds),
            "rmse": rmse
        }
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(df), index=df.index)

    def plot_predictions(self, n: int | None = None):
        preds = self.model.predict(self.X_test)
        n = len(self.X_test) if n is None else n
        idx = self.X_test.index[:n]

        plt.figure(figsize=(12, 5))
        plt.plot(idx, self.y_test.iloc[:n], label="Ground Truth", linewidth=2)
        plt.plot(idx, preds[:n], label="Prediction", linewidth=2)
        plt.title("Random Forest: Prediction vs Ground Truth")
        plt.xlabel("Index")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_predictions_plotly(self, n: int | None = None, html_path: str | None = None):
        """
        Plotly version: interactive plot of predictions vs ground truth.

        Args:
            n (int | None): Number of test samples to plot. If None, plot all.
            html_path (str | None): If provided, saves the plot as HTML to this path.
        """
        preds = self.model.predict(self.X_test)
        n = len(self.X_test) if n is None else n
        idx = self.X_test.index[:n]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx,
            y=self.y_test.iloc[:n],
            mode='lines',
            name='Ground Truth'
        ))
        fig.add_trace(go.Scatter(
            x=idx,
            y=preds[:n],
            mode='lines',
            name='Prediction'
        ))

        fig.update_layout(
            title="Random Forest: Prediction vs Ground Truth (Plotly)",
            xaxis_title="Index",
            yaxis_title=self.target_col,
            legend=dict(x=0, y=1.05, orientation="h"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        if html_path:
            fig.write_html(html_path)
        else:
            fig.show()

    def get_model(self):
        return self.model
