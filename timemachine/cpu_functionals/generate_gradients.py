import sympy as sp

x0, y0, z0 = sp.symbols("x0 y0 z0")
x1, y1, z1 = sp.symbols("x1 y1 z1")
x2, y2, z2 = sp.symbols("x2 y2 z2")

def get_dim(arg):
    if arg[0] == "x":
        return str(0)
    elif arg[0] == "y":
        return str(1)
    elif arg[0] == "z":
        return str(2)


def get_param_idx(arg):
    return arg + "_idx"

def bond_grads():

    kb, b0 = sp.symbols("kb b0")
    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    dij = sp.sqrt(dx*dx + dy*dy + dz*dz)
    db = dij - b0

    bond_nrg = kb/2*db*db


    parameters = [kb, b0]
    variables = [x0,y0,z0,x1,y1,z1]



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

            out_str = "atomicAdd(hessian_out + "+idx0+"*3*N*3 + " + dim0 + "*N*3 + " + idx1+"*3 + " + dim1 + ","
            ccode = sp.ccode(sp.diff(sp.diff(bond_nrg, v0), v1))
            print(out_str, ccode, ");")

    # rows
    for v0 in variables:
        # cols
        idx0 = get_idx(v0.name)
        dim0 = get_dim(v0.name)
        for p0 in parameters:
            p0_idx = get_param_idx(p0.name)

            out_str = "atomicAdd(mp_out + "+p0_idx+"*N*3 + "+idx0+"*3 + " + dim0 + ","
            ccode = sp.ccode(sp.diff(sp.diff(bond_nrg, v0), p0))
            print(out_str, ccode, ");")

def angle_grads():

    ka, a0 = sp.symbols("ka a0")

    vij_x = x1 - x0;
    vij_y = y1 - y0;
    vij_z = z1 - z0;

    vjk_x = x1 - x2;
    vjk_y = y1 - y2;
    vjk_z = z1 - z2;

    nij = sp.sqrt(vij_x*vij_x + vij_y*vij_y + vij_z*vij_z);
    njk = sp.sqrt(vjk_x*vjk_x + vjk_y*vjk_y + vjk_z*vjk_z);

    n3ij = nij*nij*nij;
    n3jk = njk*njk*njk;

    top = vij_x*vjk_x + vij_y*vjk_y + vij_z*vjk_z;

    ca = top/(nij*njk);

    delta = ca - sp.cos(a0)

    angle_nrg = ka/2*(delta*delta)


    variables = [x0,y0,z0,x1,y1,z1,x2,y2,z2]


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

            out_str = "atomicAdd(mp_out + "+p0_idx+"*N*3 + "+idx0+"*3 + " + dim0 + ","
            ccode = sp.ccode(sp.diff(sp.diff(angle_nrg, v0), p0))
            print(out_str, ccode, ");")


angle_grads()
