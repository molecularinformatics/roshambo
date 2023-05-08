import numpy as np

from roshambo.grid import Grid

KAPPA = 2.41798793102
PI = 3.14159265358
RT22 = 2.82842712475


def calc_hard_sphere_volume(mol):
    def gridspacetorealspace(res, lb, x, y, z):
        xi = lb[0] + (x + 0.5) * res
        yi = lb[1] + (y + 0.5) * res
        zi = lb[2] + (z + 0.5) * res
        return np.array([xi, yi, zi])

    radii_coords = mol.get_atomic_coordinates_and_radii()
    coords = radii_coords[:, :3]
    radii = radii_coords[:, 3]

    grid = Grid(mol, res=0.4, margin=0.4)
    grid.create_grid()

    volume = 0
    for x in range(grid.extent[0]):
        for y in range(grid.extent[1]):
            for z in range(grid.extent[2]):
                gridval = False
                gc = gridspacetorealspace(grid.res, grid.lb, x, y, z)
                for i in range(len(radii_coords)):
                    gridval = gridval or (np.sum((gc - coords[i]) ** 2) < radii[i] ** 2)
                if gridval:
                    volume += 1
    return volume * grid.res**3


def calc_gaussian_volume(mol):
    def gridspacetorealspace(res, lb, x, y, z):
        xi = lb[0] + (x + 0.5) * res
        yi = lb[1] + (y + 0.5) * res
        zi = lb[2] + (z + 0.5) * res
        return np.array([xi, yi, zi])

    def rho(atom, gc):
        rt22 = 2.82842712475
        partialalpha = -2.41798793102
        alpha = partialalpha / (atom[3] * atom[3])
        diffs = gc - atom[:3]
        r2 = np.sum(diffs * diffs, axis=-1)
        return rt22 * np.exp(alpha * r2)

    radii_coords = mol.get_atomic_coordinates_and_radii()

    grid = Grid(mol, res=0.4, margin=0.4)
    grid.create_grid()

    volume = 0.0
    for x in range(grid.extent[0]):
        for y in range(grid.extent[1]):
            for z in range(grid.extent[2]):
                gridval = 1.0
                gc = gridspacetorealspace(grid.res, grid.lb, x, y, z)
                for i in radii_coords:
                    gridval *= 1 - rho(i, gc)
                volume += 1 - gridval
    return volume * grid.res**3


def calc_analytic_volume(mol):
    radii_coords = mol.get_atomic_coordinates_and_radii()
    coords = radii_coords[:, :3]
    radii = radii_coords[:, 3]
    volume = np.sum((4 / 3) * PI * radii**3)
    overlap = 0
    overlap2 = 0
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            # 2nd order
            dist_sqr = np.sum((coords[i] - coords[j]) ** 2)
            pi = (4 / 3) * PI * (KAPPA / PI) ** 1.5
            pij = pi * pi
            ai = KAPPA / (radii[i] ** 2)
            aj = KAPPA / (radii[j] ** 2)
            aij = ai + aj
            dij = ai * aj * dist_sqr
            vij = pij * np.exp(-dij / aij) * (PI / aij) ** 1.5
            overlap += vij
            overlap2 += vij
            for k in range(j + 1, len(radii)):
                # 3rd order
                dist_sqr_ik = np.sum((coords[i] - coords[k]) ** 2)
                dist_sqr_jk = np.sum((coords[j] - coords[k]) ** 2)
                pijk = pij * pi
                ak = KAPPA / (radii[k] ** 2)
                aijk = aij + ak
                dik = ai * ak * dist_sqr_ik
                djk = aj * ak * dist_sqr_jk
                dijk = dij + dik + djk
                vijk = pijk * np.exp(-dijk / aijk) * (PI / aijk) ** 1.5
                overlap -= vijk
                for l in range(k + 1, len(radii)):
                    # 4th order
                    dist_sqr_il = np.sum((coords[i] - coords[l]) ** 2)
                    dist_sqr_jl = np.sum((coords[j] - coords[l]) ** 2)
                    dist_sqr_kl = np.sum((coords[k] - coords[l]) ** 2)
                    pijkl = pijk * pi
                    al = KAPPA / (radii[l] ** 2)
                    aijkl = aijk + al
                    dil = ai * al * dist_sqr_il
                    djl = aj * al * dist_sqr_jl
                    dkl = ak * al * dist_sqr_kl
                    dijkl = dijk + dil + djl + dkl
                    vijkl = pijkl * np.exp(-dijkl / aijkl) * (PI / aijkl) ** 1.5
                    overlap += vijkl
                    for m in range(l + 1, len(radii)):
                        # 5th order
                        dist_sqr_im = np.sum((coords[i] - coords[m]) ** 2)
                        dist_sqr_jm = np.sum((coords[j] - coords[m]) ** 2)
                        dist_sqr_km = np.sum((coords[k] - coords[m]) ** 2)
                        dist_sqr_lm = np.sum((coords[l] - coords[m]) ** 2)
                        pijklm = pijkl * pi
                        am = KAPPA / (radii[m] ** 2)
                        aijklm = aijkl + am
                        dim = ai * am * dist_sqr_im
                        djm = aj * am * dist_sqr_jm
                        dkm = ak * am * dist_sqr_km
                        dlm = al * am * dist_sqr_lm
                        dijklm = dijkl + dim + djm + dkm + dlm
                        vijklm = (
                            pijklm * np.exp(-dijklm / aijklm) * (PI / aijklm) ** 1.5
                        )
                        overlap -= vijklm
                        for n in range(m + 1, len(radii)):
                            # 6th order
                            dist_sqr_in = np.sum((coords[i] - coords[n]) ** 2)
                            dist_sqr_jn = np.sum((coords[j] - coords[n]) ** 2)
                            dist_sqr_kn = np.sum((coords[k] - coords[n]) ** 2)
                            dist_sqr_ln = np.sum((coords[l] - coords[n]) ** 2)
                            dist_sqr_mn = np.sum((coords[m] - coords[n]) ** 2)
                            pijklmn = pijklm * pi
                            an = KAPPA / (radii[n] ** 2)
                            aijklmn = aijklm + an
                            din = ai * an * dist_sqr_in
                            djn = aj * an * dist_sqr_jn
                            dkn = ak * an * dist_sqr_kn
                            dln = al * an * dist_sqr_ln
                            dmn = am * an * dist_sqr_mn
                            dijklmn = dijklm + din + djn + dkn + dln + dmn
                            vijklmn = (
                                pijklmn
                                * np.exp(-dijklmn / aijklmn)
                                * (PI / aijklmn) ** 1.5
                            )
                            overlap += vijklmn
    return volume - overlap, volume - overlap2


def calc_analytic_volume_cutoff(mol, epsilon=0.1):
    radii_coords = mol.get_atomic_coordinates_and_radii()
    coords = radii_coords[:, :3]
    radii = radii_coords[:, 3]
    volume = np.sum((4 / 3) * PI * radii**3)
    overlap = 0
    overlap2 = 0
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            # 2nd order
            if np.linalg.norm(coords[i] - coords[j]) <= radii[i] + radii[j] + epsilon:
                dist_sqr = np.sum((coords[i] - coords[j]) ** 2)
                pi = (4 / 3) * PI * (KAPPA / PI) ** 1.5
                pij = pi * pi
                ai = KAPPA / (radii[i] ** 2)
                aj = KAPPA / (radii[j] ** 2)
                aij = ai + aj
                dij = ai * aj * dist_sqr
                vij = pij * np.exp(-dij / aij) * (PI / aij) ** 1.5
                overlap += vij
                overlap2 += vij
                for k in range(j + 1, len(radii)):
                    # 3rd order
                    if (
                        np.linalg.norm(coords[i] - coords[k])
                        <= radii[i] + radii[k] + epsilon
                        and np.linalg.norm(coords[j] - coords[k])
                        <= radii[j] + radii[k] + epsilon
                    ):
                        dist_sqr_ik = np.sum((coords[i] - coords[k]) ** 2)
                        dist_sqr_jk = np.sum((coords[j] - coords[k]) ** 2)
                        pijk = pij * pi
                        ak = KAPPA / (radii[k] ** 2)
                        aijk = aij + ak
                        dik = ai * ak * dist_sqr_ik
                        djk = aj * ak * dist_sqr_jk
                        dijk = dij + dik + djk
                        vijk = pijk * np.exp(-dijk / aijk) * (PI / aijk) ** 1.5
                        overlap -= vijk
                        for l in range(k + 1, len(radii)):
                            # 4th order
                            if (
                                np.linalg.norm(coords[i] - coords[l])
                                <= radii[i] + radii[l] + epsilon
                                and np.linalg.norm(coords[j] - coords[l])
                                <= radii[j] + radii[l] + epsilon
                                and np.linalg.norm(coords[k] - coords[l])
                                <= radii[k] + radii[l] + epsilon
                            ):
                                dist_sqr_il = np.sum((coords[i] - coords[l]) ** 2)
                                dist_sqr_jl = np.sum((coords[j] - coords[l]) ** 2)
                                dist_sqr_kl = np.sum((coords[k] - coords[l]) ** 2)
                                pijkl = pijk * pi
                                al = KAPPA / (radii[l] ** 2)
                                aijkl = aijk + al
                                dil = ai * al * dist_sqr_il
                                djl = aj * al * dist_sqr_jl
                                dkl = ak * al * dist_sqr_kl
                                dijkl = dijk + dil + djl + dkl
                                vijkl = (
                                    pijkl * np.exp(-dijkl / aijkl) * (PI / aijkl) ** 1.5
                                )
                                overlap += vijkl
                                for m in range(l + 1, len(radii)):
                                    # 5th order
                                    if (
                                        np.linalg.norm(coords[i] - coords[m])
                                        <= radii[i] + radii[m] + epsilon
                                        and np.linalg.norm(coords[j] - coords[m])
                                        <= radii[j] + radii[m] + epsilon
                                        and np.linalg.norm(coords[k] - coords[m])
                                        <= radii[k] + radii[m] + epsilon
                                        and np.linalg.norm(coords[l] - coords[m])
                                        <= radii[l] + radii[m] + epsilon
                                    ):
                                        dist_sqr_im = np.sum(
                                            (coords[i] - coords[m]) ** 2
                                        )
                                        dist_sqr_jm = np.sum(
                                            (coords[j] - coords[m]) ** 2
                                        )
                                        dist_sqr_km = np.sum(
                                            (coords[k] - coords[m]) ** 2
                                        )
                                        dist_sqr_lm = np.sum(
                                            (coords[l] - coords[m]) ** 2
                                        )
                                        pijklm = pijkl * pi
                                        am = KAPPA / (radii[m] ** 2)
                                        aijklm = aijkl + am
                                        dim = ai * am * dist_sqr_im
                                        djm = aj * am * dist_sqr_jm
                                        dkm = ak * am * dist_sqr_km
                                        dlm = al * am * dist_sqr_lm
                                        dijklm = dijkl + dim + djm + dkm + dlm
                                        vijklm = (
                                            pijklm
                                            * np.exp(-dijklm / aijklm)
                                            * (PI / aijklm) ** 1.5
                                        )
                                        overlap -= vijklm
                                        for n in range(m + 1, len(radii)):
                                            # 6th order
                                            if (
                                                np.linalg.norm(coords[i] - coords[n])
                                                <= radii[i] + radii[n] + epsilon
                                                and np.linalg.norm(
                                                    coords[j] - coords[n]
                                                )
                                                <= radii[j] + radii[n] + epsilon
                                                and np.linalg.norm(
                                                    coords[k] - coords[n]
                                                )
                                                <= radii[k] + radii[n] + epsilon
                                                and np.linalg.norm(
                                                    coords[l] - coords[n]
                                                )
                                                <= radii[l] + radii[n] + epsilon
                                                and np.linalg.norm(
                                                    coords[m] - coords[n]
                                                )
                                                <= radii[m] + radii[n] + epsilon
                                            ):
                                                dist_sqr_in = np.sum(
                                                    (coords[i] - coords[n]) ** 2
                                                )
                                                dist_sqr_jn = np.sum(
                                                    (coords[j] - coords[n]) ** 2
                                                )
                                                dist_sqr_kn = np.sum(
                                                    (coords[k] - coords[n]) ** 2
                                                )
                                                dist_sqr_ln = np.sum(
                                                    (coords[l] - coords[n]) ** 2
                                                )
                                                dist_sqr_mn = np.sum(
                                                    (coords[m] - coords[n]) ** 2
                                                )
                                                pijklmn = pijklm * pi
                                                an = KAPPA / (radii[n] ** 2)
                                                aijklmn = aijklm + an
                                                din = ai * an * dist_sqr_in
                                                djn = aj * an * dist_sqr_jn
                                                dkn = ak * an * dist_sqr_kn
                                                dln = al * an * dist_sqr_ln
                                                dmn = am * an * dist_sqr_mn
                                                dijklmn = (
                                                    dijklm + din + djn + dkn + dln + dmn
                                                )
                                                vijklmn = (
                                                    pijklmn
                                                    * np.exp(-dijklmn / aijklmn)
                                                    * (PI / aijklmn) ** 1.5
                                                )
                                                overlap += vijklmn
    return volume - overlap, volume - overlap2
