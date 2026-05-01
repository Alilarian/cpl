"""
Soft-optimal advantage for PointMassGymEnv via discrete value iteration.

Grid design
-----------
State  : N × N Cartesian cell-center grid over [0,1]²
         Default N=100 → cell width = max_speed = 0.01 (one step per cell)
Action : n_act × n_act Cartesian grid over [-1,1]²
         Default n_act=11 → 121 actions, spacing 0.2 — matches SAC convention exactly

Key design choices
------------------
- Cartesian actions: no polar conversion needed; directly comparable to SAC outputs.
- All-or-nothing wall: if either axis would leave [0,1), agent stays put — exact
  replication of env.step() line `if (new_pos >= 0.0).all() and (new_pos < 1.0).all()`.
- Terminal cells are absorbing: V = 0 zeroed twice per iteration (before and after
  the logsumexp backup) to prevent value leakage into/out of terminal regions.
- Soft Bellman backup: V = α · logsumexp(Q/α) gives the MaxEnt soft-optimal value
  consistent with π*(a|s) ∝ exp(A*(s,a)/α) — the distribution CPL assumes.
- Advantage query: bilinear interpolation over (x,y) state, nearest-neighbor over action.
"""

import numpy as np


class AdvantageVI:
    """
    Precomputes V*, Q*, A* for PointMassGymEnv via soft Bellman value iteration.

    Usage
    -----
    from research.envs.pointmass import PointMassGymEnv
    env = PointMassGymEnv()
    avi = AdvantageVI(env, alpha=0.05).build()
    avi.save("runs/pm_sac_oracle/advantage_vi.npz")

    # Later, score a trajectory segment:
    avi2 = AdvantageVI.load("runs/pm_sac_oracle/advantage_vi.npz")
    score = avi2.segment_advantage(obses, actions)   # (T,2) arrays
    """

    def __init__(
        self,
        env,
        alpha: float = 0.1,
        N: int = 100,
        n_act: int = 11,
        gamma: float = 0.99,
        tol: float = 1e-6,
        max_iter: int = 3000,
    ):
        """
        Parameters
        ----------
        env     : PointMassGymEnv instance (provides goal/trap/thresholds/max_speed)
        alpha   : SAC temperature; use the learned log_alpha.exp() from the checkpoint
        N       : state-grid side length (N² total states)
        n_act   : action-grid points per axis (n_act² total actions)
        gamma   : discount factor — must match SAC training
        tol     : convergence threshold on max|ΔV|
        max_iter: safety cap on VI iterations
        """
        self.env = env
        self.alpha = float(alpha)
        self.N = N
        self.n_act = n_act
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

        cell = 1.0 / N
        self.xs = np.linspace(cell / 2.0, 1.0 - cell / 2.0, N)  # cell centres

        ax = np.linspace(-1.0, 1.0, n_act)
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        self.action_grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        # Populated by build()
        self.V: np.ndarray = None
        self.Q: np.ndarray = None
        self.A_star: np.ndarray = None
        self._terminal: np.ndarray = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, verbose: bool = True) -> "AdvantageVI":
        """Precompute T, R, then run VI. Returns self for chaining."""
        K = len(self.action_grid)
        if verbose:
            print(
                f"[VI] {self.N}×{self.N} states  {K} actions  "
                f"α={self.alpha}  γ={self.gamma}"
            )

        T, R, terminal = self._build_tables()
        self._terminal = terminal

        if verbose:
            n_term = int(terminal.sum())
            pct = 100.0 * n_term / self.N ** 2
            print(f"[VI] terminal cells: {n_term} ({pct:.1f}%)")
            print("[VI] running soft Bellman VI …")

        V = self._run_vi(T, R, terminal, verbose=verbose)
        self.V = V

        # Recompute Q and A with converged V
        V_next = V[T[:, :, :, 0], T[:, :, :, 1]].copy()  # (N, N, K)
        V_next[terminal] = 0.0
        Q = R.astype(np.float64) + self.gamma * V_next
        A = Q - V[:, :, np.newaxis]
        A[terminal] = 0.0

        self.Q = Q.astype(np.float32)
        self.A_star = A.astype(np.float32)
        return self

    def _build_tables(self):
        """
        Returns
        -------
        T        : (N, N, K, 2) int16   — next-state grid indices
        R        : (N, N, K)    float32 — immediate reward
        terminal : (N, N)       bool    — cells whose centre is inside goal or trap
        """
        N, xs, env = self.N, self.xs, self.env
        K = len(self.action_grid)
        cell = float(xs[1] - xs[0])

        XX, YY = np.meshgrid(xs, xs, indexing="ij")
        pos_grid = np.stack([XX, YY], axis=-1)       # (N, N, 2)
        pos_flat = pos_grid.reshape(-1, 2)            # (N², 2)

        # Terminal mask — based on cell-centre distances
        dist_goal = np.linalg.norm(pos_grid - env.goal, axis=-1)
        dist_trap = np.linalg.norm(pos_grid - env.trap, axis=-1)
        terminal = (
            (dist_goal <= env.goal_dist_thresh)
            | (dist_trap <= env.trap_dist_thresh)
        )

        # Fully vectorised over all (state, action) pairs
        # velocity[k] = action_grid[k] * max_speed  — Cartesian, matches env.step()
        velocities = (self.action_grid * env.max_speed).astype(np.float64)  # (K, 2)

        # new_pos_all[k, n] = pos_flat[n] + velocities[k]  → shape (K, N², 2)
        new_pos_all = pos_flat[np.newaxis] + velocities[:, np.newaxis]  # (K, N², 2)

        # All-or-nothing boundary: if either axis leaves [0,1), stay put
        in_bounds = (
            (new_pos_all >= 0.0).all(axis=2) & (new_pos_all < 1.0).all(axis=2)
        )  # (K, N²)
        effective = np.where(
            in_bounds[:, :, np.newaxis], new_pos_all, pos_flat[np.newaxis]
        )  # (K, N², 2)

        # Snap to grid index via floor division
        idx = np.clip(
            np.floor(effective / cell).astype(int), 0, N - 1
        )  # (K, N², 2)

        # Rearrange axes: (K, N², *) → (N, N, K, *)
        T = np.stack(
            [
                idx[:, :, 0].reshape(K, N, N).transpose(1, 2, 0),
                idx[:, :, 1].reshape(K, N, N).transpose(1, 2, 0),
            ],
            axis=3,
        ).astype(np.int16)  # (N, N, K, 2)

        # Reward at next cell centre  (analytic terminal reward + step cost)
        next_xs = xs[idx[:, :, 0]]  # (K, N²)
        next_ys = xs[idx[:, :, 1]]  # (K, N²)
        next_centers = np.stack([next_xs, next_ys], axis=2)  # (K, N², 2)

        dg = np.linalg.norm(next_centers - env.goal, axis=2)   # (K, N²)
        dt = np.linalg.norm(next_centers - env.trap, axis=2)   # (K, N²)
        at_goal = (dg <= env.goal_dist_thresh).astype(np.float32)
        at_trap = (dt <= env.trap_dist_thresh).astype(np.float32)

        r = (
            env.succ_rew_bonus * at_goal
            + env.crash_rew_penalty * at_trap
            - env.step_cost
        )  # (K, N²)
        R = r.reshape(K, N, N).transpose(1, 2, 0).astype(np.float32)  # (N, N, K)

        return T, R, terminal

    def _run_vi(self, T, R, terminal, verbose: bool = True) -> np.ndarray:
        alpha, gamma = self.alpha, self.gamma
        V = np.zeros((self.N, self.N), dtype=np.float64)
        delta = float("inf")

        for it in range(self.max_iter):
            V_next = V[T[:, :, :, 0], T[:, :, :, 1]].copy()  # (N, N, K)
            V_next[terminal] = 0.0                             # absorbing: zeroed before backup

            Q = R.astype(np.float64) + gamma * V_next          # (N, N, K)

            # Numerically stable logsumexp over action axis
            q = Q / alpha
            q_max = q.max(axis=2, keepdims=True)
            V_new = alpha * (
                np.log(np.exp(q - q_max).sum(axis=2)) + q_max.squeeze(2)
            )
            V_new[terminal] = 0.0                              # absorbing: zeroed after backup

            delta = float(np.abs(V_new - V).max())
            V = V_new

            if verbose and (it + 1) % 200 == 0:
                print(f"  iter {it + 1:4d}  ΔV={delta:.2e}")

            if delta < self.tol:
                if verbose:
                    print(f"[VI] converged at iter {it + 1}  ΔV={delta:.2e}")
                break
        else:
            if verbose:
                print(
                    f"[VI] Warning: did not converge after {self.max_iter} iters  "
                    f"ΔV={delta:.2e}"
                )

        return V

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _bilinear_weights(self, obs: np.ndarray):
        """Map continuous position to surrounding grid corners + fractional weights."""
        cell = float(self.xs[1] - self.xs[0])
        # Fractional position in index space: 0.0 → centre of cell 0, 1.0 → centre of cell 1
        fx = float(obs[0]) / cell - 0.5
        fy = float(obs[1]) / cell - 0.5
        i0 = int(np.clip(int(fx), 0, self.N - 2))
        j0 = int(np.clip(int(fy), 0, self.N - 2))
        wx = float(np.clip(fx - i0, 0.0, 1.0))
        wy = float(np.clip(fy - j0, 0.0, 1.0))
        return i0, j0, i0 + 1, j0 + 1, wx, wy

    def query(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Return A*(s, a) for a single continuous (obs, action) pair."""
        assert self.A_star is not None, "call build() first"
        i0, j0, i1, j1, wx, wy = self._bilinear_weights(obs)
        k = int(np.argmin(np.linalg.norm(self.action_grid - action, axis=1)))
        return float(
            self.A_star[i0, j0, k] * (1 - wx) * (1 - wy)
            + self.A_star[i1, j0, k] * wx * (1 - wy)
            + self.A_star[i0, j1, k] * (1 - wx) * wy
            + self.A_star[i1, j1, k] * wx * wy
        )

    def query_batch(self, obses: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Vectorized A*(s, a) lookup.

        Parameters
        ----------
        obses   : (T, 2) float  — continuous positions in [0,1]²
        actions : (T, 2) float  — Cartesian actions in [-1,1]²

        Returns
        -------
        (T,) float  — per-step soft advantages
        """
        assert self.A_star is not None, "call build() first"
        cell = float(self.xs[1] - self.xs[0])

        fx = obses[:, 0] / cell - 0.5
        fy = obses[:, 1] / cell - 0.5
        i0 = np.clip(fx.astype(int), 0, self.N - 2)
        j0 = np.clip(fy.astype(int), 0, self.N - 2)
        wx = np.clip(fx - i0, 0.0, 1.0)
        wy = np.clip(fy - j0, 0.0, 1.0)
        i1, j1 = i0 + 1, j0 + 1

        # Nearest-neighbor action lookup (T, K, 2) → (T,)
        diffs = self.action_grid[np.newaxis] - actions[:, np.newaxis]  # (T, K, 2)
        k = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)           # (T,)

        a00 = self.A_star[i0, j0, k]
        a10 = self.A_star[i1, j0, k]
        a01 = self.A_star[i0, j1, k]
        a11 = self.A_star[i1, j1, k]

        return (
            a00 * (1 - wx) * (1 - wy)
            + a10 * wx * (1 - wy)
            + a01 * (1 - wx) * wy
            + a11 * wx * wy
        )

    def segment_advantage(
        self,
        obses: np.ndarray,
        actions: np.ndarray,
        gamma: float = None,
    ) -> float:
        """
        Discounted sum of per-step advantages over a trajectory segment.

            score(σ) = Σ_t  γ^t · A*(s_t, a_t)

        Parameters
        ----------
        obses   : (T, 2) — positions along the segment
        actions : (T, 2) — actions taken at each step
        gamma   : discount; defaults to self.gamma

        Returns
        -------
        float — segment advantage score (higher = more preferred)
        """
        if gamma is None:
            gamma = self.gamma
        adv = self.query_batch(
            np.asarray(obses, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
        )
        t = np.arange(len(adv), dtype=np.float64)
        return float((gamma ** t * adv).sum())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        assert self.A_star is not None, "call build() first"
        np.savez_compressed(
            path,
            xs=self.xs,
            action_grid=self.action_grid,
            V=self.V.astype(np.float32),
            Q=self.Q,
            A_star=self.A_star,
            terminal=self._terminal,
            # Scalar metadata packed into one array for portability
            meta=np.array([self.alpha, self.gamma, float(self.N), float(self.n_act)]),
        )
        print(f"[VI] saved → {path}")

    @classmethod
    def load(cls, path: str, env=None) -> "AdvantageVI":
        """Load a precomputed table.  env is optional (only needed for build())."""
        data = np.load(path)
        alpha, gamma, N, n_act = data["meta"]
        obj = cls.__new__(cls)
        obj.alpha = float(alpha)
        obj.gamma = float(gamma)
        obj.N = int(N)
        obj.n_act = int(n_act)
        obj.tol = 1e-6
        obj.max_iter = 3000
        obj.env = env
        obj.xs = data["xs"]
        obj.action_grid = data["action_grid"]
        obj.V = data["V"]
        obj.Q = data["Q"]
        obj.A_star = data["A_star"]
        obj._terminal = data["terminal"]
        return obj
