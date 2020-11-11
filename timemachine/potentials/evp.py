# eigenvalue problem solvers
import numpy as np

DBL_EPSILON = 2.2204460492503131e-16

def dsyevc3(A):

    de = A[0][1] * A[1][2]
    dd = np.square(A[0][1])
    ee = np.square(A[1][2])
    ff = np.square(A[0][2])
    m  = A[0][0] + A[1][1] + A[2][2]
    c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2]) - (dd + ee + ff)
    c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2] - 2.0 * A[0][2]*de

    p = np.square(m) - 3.0*c1;
    q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
    sqrt_p = np.sqrt(np.fabs(p));

    phi = 27.0 * ( 0.25*np.square(c1)*(p - c1) + c0*(q + 27.0/4.0*c0))
    phi = (1.0/3.0) * np.arctan2(np.sqrt(np.fabs(phi)), q)

    c = sqrt_p*np.cos(phi);
    s = (1.0/np.sqrt(3))*sqrt_p*np.sin(phi);

    w = np.zeros(3)

    w[1]  = (1.0/3.0)*(m - c);
    w[2]  = w[1] + s;
    w[0]  = w[1] + c;
    w[1] -= s;



    return np.sort(w);
    # return w;


def dsyevv3(input_tensor):

  A = np.asarray(input_tensor).copy()

  # # Calculate eigenvalues
  w = dsyevc3(A);
  Q = np.zeros((3, 3)) # column eigenvectors

#ifndef EVALS_ONLY
  wmax = np.fabs(w[0])
  if np.fabs(w[1]) > wmax:
    wmax = np.fabs(w[1])
  if np.fabs(w[2]) > wmax:
    wmax = np.fabs(w[2])
  thresh = np.square(8.0 * DBL_EPSILON * wmax);

  # # Prepare calculation of eigenvectors
  n0tmp   = np.square(A[0][1]) + np.square(A[0][2]);
  n1tmp   = np.square(A[0][1]) + np.square(A[1][2]);
  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
  Q[2][1] = np.square(A[0][1]);

  # # Calculate first eigenvector by the formula
  # #   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
  norm    = np.square(Q[0][0]) + np.square(Q[1][0]) + np.square(Q[2][0]);
  n0      = n0tmp + np.square(A[0][0]);
  n1      = n1tmp + np.square(A[1][1]);
  error   = n0 * n1;
  
  if n0 <= thresh:       # If the first column is zero, then (1,0,0) is an eigenvector
    Q[0][0] = 1.0;
    Q[1][0] = 0.0;
    Q[2][0] = 0.0;
  elif n1 <= thresh:  # If the second column is zero, then (0,1,0) is an eigenvector
    Q[0][0] = 0.0;
    Q[1][0] = 1.0;
    Q[2][0] = 0.0;
  elif norm < np.square(64.0 * DBL_EPSILON) * error: # If angle between A[0] and A[1] is too small, don't use
    assert 0
    t = np.square(A[0][1]);       # cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];

    if np.square(A[1][1]) > t:
      t = np.square(A[1][1]);
      f = -A[0][1] / A[1][1];

    if np.square(A[1][2]) > t:
      f = -A[0][2] / A[1][2];
    norm    = 1.0/np.sqrt(1 + np.square(f));
    Q[0][0] = norm;
    Q[1][0] = f * norm;
    Q[2][0] = 0.0;
  else:                      # This is the standard branch
    norm = np.sqrt(1.0 / norm);
    # for (j=0; j < 3; j++)
    for j in range(3):
      Q[j][0] = Q[j][0] * norm;

  
  # Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if np.fabs(t) > 8.0 * DBL_EPSILON * wmax:
    # For non-degenerate eigenvalue, calculate second eigenvector by the formula
    #   v[1] = (A - w[1]).e1 x (A - w[1]).e2
    A[0][0] += t;
    A[1][1] += t;
    Q[0][1]  = Q[0][1] + A[0][2]*w[1];
    Q[1][1]  = Q[1][1] + A[1][2]*w[1];
    Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
    norm     = np.square(Q[0][1]) + np.square(Q[1][1]) + np.square(Q[2][1]);
    n0       = n0tmp + np.square(A[0][0]);
    n1       = n1tmp + np.square(A[1][1]);
    error    = n0 * n1;
 
    if n0 <= thresh:       # If the first column is zero, then (1,0,0) is an eigenvector
      Q[0][1] = 1.0;
      Q[1][1] = 0.0;
      Q[2][1] = 0.0;
    elif n1 <= thresh:  # If the second column is zero, then (0,1,0) is an eigenvector
      Q[0][1] = 0.0;
      Q[1][1] = 1.0;
      Q[2][1] = 0.0;
    elif norm < np.square(64.0 * DBL_EPSILON) * error:
      t = np.square(A[0][1]);     # cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if np.square(A[1][1]) > t:
        t = np.square(A[1][1]);
        f = -A[0][1] / A[1][1];
      if np.square(A[1][2]) > t:
        f = -A[0][2] / A[1][2];
      norm    = 1.0/np.sqrt(1 + np.square(f));
      Q[0][1] = norm;
      Q[1][1] = f * norm;
      Q[2][1] = 0.0;
    else:
      norm = np.sqrt(1.0 / norm);
      for j in range(3):
        Q[j][1] = Q[j][1] * norm;

  else:
    assert 0
  
  # Calculate third eigenvector according to
  #   v[2] = v[0] x v[1]
  Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];

  for d in range(3):
    np.testing.assert_almost_equal(
      np.matmul(input_tensor, Q[:, d]),
      w[d]*Q[:, d]
    )

  return w, Q
