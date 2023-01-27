class Grid:
    def __init__(self, mol, res, margin):
        self.mol = mol
        self.res = res
        self.margin = margin

    def get_bounding_box(self):
        lb = [float("inf"), float("inf"), float("inf")]
        ub = [float("-inf"), float("-inf"), float("-inf")]
        for i in range(self.mol.natoms):
            lb[0] = min(lb[0], self.mol.atoms[i].x)
            lb[1] = min(lb[1], self.mol.atoms[i].y)
            lb[2] = min(lb[2], self.mol.atoms[i].z)
            ub[0] = max(ub[0], self.mol.atoms[i].x)
            ub[1] = max(ub[1], self.mol.atoms[i].y)
            ub[2] = max(ub[2], self.mol.atoms[i].z)
        lb = [x - self.margin for x in lb]
        ub = [x + self.margin for x in ub]
        return lb, ub

    def get_host_grid(self):
        pass
