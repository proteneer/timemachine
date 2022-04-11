import sympy as sp

x0, y0, z0, w0 = sp.symbols("x0 y0 z0 w0")
x1, y1, z1, w1 = sp.symbols("x1 y1 z1 w1")
x2, y2, z2, w2 = sp.symbols("x2 y2 z2 w2")
x3, y3, z3, w3 = sp.symbols("x3 y3 z3 w3")

NDIMS = 4


def get_dim(arg):
    if arg[0] == "x":
        return str(0)
    elif arg[0] == "y":
        return str(1)
    elif arg[0] == "z":
        return str(2)
    elif arg[0] == "w":
        return str(3)


def get_param_idx(arg):
    return arg + "_idx"


def bond_grads():

    kb, b0 = sp.symbols("kb b0")
    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    dw = w0 - w1

    dij = sp.sqrt(dx * dx + dy * dy + dz * dz + dw * dw)
    db = dij - b0

    bond_nrg = kb / 2 * db * db

    parameters = [kb, b0]
    variables = [x0, y0, z0, w0, x1, y1, z1, w1]

    def get_idx(arg):
        if arg[1] == "0":
            return "src_idx"
        elif arg[1] == "1":
            return "dst_idx"

    # rows
    for v0 in variables:
        # cols
        idx0 = get_idx(v0.name)
        dim0 = get_dim(v0.name)
        for v1 in variables:
            idx1 = get_idx(v1.name)
            dim1 = get_dim(v1.name)

            out_str = (
                "atomicAdd(d2E_dx2 + conf_idx*N*NDIMS*N*NDIMS + "
                + idx0
                + "*NDIMS*N*NDIMS + "
                + dim0
                + "*N*NDIMS + "
                + idx1
                + "*NDIMS + "
                + dim1
                + ","
            )
            ccode = sp.ccode(sp.diff(sp.diff(bond_nrg, v0), v1))
            print(out_str, ccode, ");")

    # rows
    for v0 in variables:
        # cols
        idx0 = get_idx(v0.name)
        dim0 = get_dim(v0.name)
        for p0 in parameters:
            p0_idx = get_param_idx(p0.name)

            out_str = "atomicAdd(mp_out + " + p0_idx + "*N*3 + " + idx0 + "*3 + " + dim0 + ","
            ccode = sp.ccode(sp.diff(sp.diff(bond_nrg, v0), p0))
            print(out_str, ccode, ");")


def angle_grads():

    ka, a0 = sp.symbols("ka a0")

    vij_x = x1 - x0
    vij_y = y1 - y0
    vij_z = z1 - z0

    vjk_x = x1 - x2
    vjk_y = y1 - y2
    vjk_z = z1 - z2

    nij = sp.sqrt(vij_x * vij_x + vij_y * vij_y + vij_z * vij_z)
    njk = sp.sqrt(vjk_x * vjk_x + vjk_y * vjk_y + vjk_z * vjk_z)

    n3ij = nij * nij * nij
    n3jk = njk * njk * njk

    top = vij_x * vjk_x + vij_y * vjk_y + vij_z * vjk_z

    ca = top / (nij * njk)

    delta = ca - sp.cos(a0)

    angle_nrg = ka / 2 * (delta * delta)

    variables = [x0, y0, z0, x1, y1, z1, x2, y2, z2]

    parameters = [ka, a0]

    def get_idx(arg):
        if arg[1] == "0":
            return "atom_0_idx"
        elif arg[1] == "1":
            return "atom_1_idx"
        elif arg[1] == "2":
            return "atom_2_idx"

    # rows
    # for v0 in variables:
    #     # cols
    #     idx0 = get_idx(v0.name)
    #     dim0 = get_dim(v0.name)
    #     for v1 in variables:
    #         idx1 = get_idx(v1.name)
    #         dim1 = get_dim(v1.name)

    #         out_str = "atomicAdd(hessian_out + "+idx0+"*3*N*3 + " + dim0 + "*N*3 + " + idx1+"*3 + " + dim1 + ","
    #         ccode = sp.ccode(sp.diff(sp.diff(angle_nrg, v0), v1))
    #         print(out_str, ccode, ");\n")

    for v0 in variables:
        # cols
        idx0 = get_idx(v0.name)
        dim0 = get_dim(v0.name)
        for p0 in parameters:
            p0_idx = get_param_idx(p0.name)

            out_str = "atomicAdd(mp_out + " + p0_idx + "*N*3 + " + idx0 + "*3 + " + dim0 + ","
            ccode = sp.ccode(sp.diff(sp.diff(angle_nrg, v0), p0))
            print(out_str, ccode, ");")


def cross_product(a1, a2, a3, b1, b2, b3):

    s1 = a2 * b3 - a3 * b2
    s2 = a3 * b1 - a1 * b3
    s3 = a1 * b2 - a2 * b1

    return s1, s2, s3


def norm(x, y, z):
    return sp.sqrt(x * x + y * y + z * z)


def dot_product(x0, y0, z0, x1, y1, z1):
    return x0 * x1 + y0 * y1 + z0 * z1


def torsion_grads():

    k, period, phase = sp.symbols("k period phase")

    rij_x = x0 - x1
    rij_y = y0 - y1
    rij_z = z0 - z1

    rkj_x = x2 - x1
    rkj_y = y2 - y1
    rkj_z = z2 - z1

    rkl_x = x2 - x3
    rkl_y = y2 - y3
    rkl_z = z2 - z3

    n1_x, n1_y, n1_z = cross_product(rij_x, rij_y, rij_z, rkj_x, rkj_y, rkj_z)
    n2_x, n2_y, n2_z = cross_product(rkj_x, rkj_y, rkj_z, rkl_x, rkl_y, rkl_z)
    n3_x, n3_y, n3_z = cross_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z)

    rkj_n = norm(rkj_x, rkj_y, rkj_z)

    rkj_x /= rkj_n
    rkj_y /= rkj_n
    rkj_z /= rkj_n

    y = dot_product(n3_x, n3_y, n3_z, rkj_x, rkj_y, rkj_z)
    x = dot_product(n1_x, n1_y, n1_z, n2_x, n2_y, n2_z)
    angle = sp.atan2(y, x)

    nrg = k * (1 + sp.cos(period * angle - phase))

    variables = [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3]

    parameters = [k, period, phase]

    def get_idx(arg):
        if arg[1] == "0":
            return "atom_0_idx"
        elif arg[1] == "1":
            return "atom_1_idx"
        elif arg[1] == "2":
            return "atom_2_idx"
        elif arg[1] == "3":
            return "atom_3_idx"

    # # rows
    # for v0 in variables:
    #     # cols
    #     idx0 = get_idx(v0.name)
    #     dim0 = get_dim(v0.name)
    #     for v1 in variables:
    #         idx1 = get_idx(v1.name)
    #         dim1 = get_dim(v1.name)

    #         out_str = "atomicAdd(hessian_out + "+idx0+"*3*N*3 + " + dim0 + "*N*3 + " + idx1+"*3 + " + dim1 + ","
    #         ccode = sp.ccode(sp.diff(sp.diff(nrg, v0), v1))
    #         print(out_str, ccode, ");\n")

    for v0 in variables:
        # cols
        idx0 = get_idx(v0.name)
        dim0 = get_dim(v0.name)
        for p0 in parameters:
            p0_idx = get_param_idx(p0.name)

            out_str = "atomicAdd(mp_out + " + p0_idx + "*N*3 + " + idx0 + "*3 + " + dim0 + ","
            ccode = sp.ccode(sp.diff(sp.diff(nrg, v0), p0))
            print(out_str, ccode, ");")


# torsion_grads()
bond_grads()
