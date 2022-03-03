using LinearAlgebra



function Givens_Rotation(x::Vector)                     # Rotates [a;b] to [r;0]
    a = x[1]; b = x[2];c = 0.0;s = 0.0;r = 0.0;
    if (b == 0.0)
        c = sign(a)
        s = 0.0
        r = abs(a)
    elseif (a == 0.0)
        c = 0.0
        s = -sign(b)
        r = abs(b)
    elseif (abs(a) .> abs(b))
        t = b / a
        u = sign(a) * abs(sqrt(1 + t^2))
        c = 1 / u
        s = -c * t
        r = a * u
    else
        t = a / b
        u = sign(b) * abs(sqrt(1 + t^2))
        s = -1 / u
        c = -s * t
        r = b * u
    end
    return ([c -s; s c], [r; 0])#(c, s, r)
end

#%----------------------------------------------------%
#%                  Backsubstitution
#%----------------------------------------------------%

function backsubstitution(U, c)                       # For the upper triagnllaur Matrix U,
                                                      # returns the answer to Ux = c via backsubstitution algorithm
    m = size(U, 1)                                    # number of rows
    x = similar(c, typeof(c[1] / U[1, 1]))            # allocate the solution vector
    for i = m:-1:1                                    # loop over the rows from bottom to top
        r = c[i]
        for k = i+1:m
            r = r - U[i, k] * x[k]
        end
        x[i] = r / U[i, i]
    end
    return x
end


#%----------------------------------------------------%
#%                 Standard GMRES
#%----------------------------------------------------%
function reg_gmres(A::Matrix, b::Matrix, x, max_iter)
    ts = size(A, 1)
    r = b - A * x
    r_norm = norm(r)
    r_normS = [r_norm]
    Q = r / r_norm
    b_norm = norm(b)
    sn = zeros(1, 0)
    cs = zeros(1, 0)
    e1 = zeros(max_iter + 1, 1)
    e1[1] = 1.0
    beta = r_norm * e1
    H = zeros(1, 0)
    zrec = x

    for k = 1:max_iter
        (hk, qk1) = arnoldi(A, Q, k)
        Q = [Q qk1]
        (hk, cs_k, sn_k) = apply_givens_rotation(hk, cs, sn, k)
        cs = [cs cs_k]
        sn = [sn sn_k]
        H = [[H; zeros(1, k - 1)] hk]
        #update the residual vector
        beta[k+1] = -sn[k] * beta[k]
        beta[k] = cs[k] * beta[k]
        residual = abs(beta[k+1])
        r_normS = [r_normS; residual]
        #calculate the result
        λ = H \ beta[1:k+1]
        zk = x + Q[:, 1:k] * λ
        zrec = [zrec zk]

        if (residual <= 1e-6)
            break
        end
    end
    #return x, zrec, r_normS
    return x, r_normS, zrec
end

#%----------------------------------------------------%
#%               Standard Arnoldi
#%----------------------------------------------------%
function arnoldi(A::Matrix, Q::Matrix, k::Number)
    q = A * Q[:, k] #Krylov Vector
    h = zeros(k + 1, 1)
    for i = 1:k                                   #Modified Gram-Schmidt, keeping the Hessenberg matrix
        h[i] = q' * Q[:, i]
        q = q - h[i] * Q[:, i]
    end
    h[k+1] = norm(q)
    q = q / h[k+1]
    return (h, q)
end

#%---------------------------------------------------------------------%
#%                  Applying Givens Rotation to H col                  %
#%---------------------------------------------------------------------%
function apply_givens_rotation(h, cs, sn, k)
                                                   #apply for ith row
    for i = 1:k-1
        temp = cs[i] * h[i] + sn[i] * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i] = temp
    end
    (cs_k, sn_k) = givens_rotation(h[k], h[k+1])   #update the next sin cos values for rotation
    h[k] = cs_k * h[k] + sn_k * h[k+1]             #eliminate H(i + 1, i)
    h[k+1] = 0.0
    return (h, cs_k, sn_k)
end

                                                   #Calculate the Given rotation matrix
function givens_rotation(v1, v2)
    t = sqrt(v1^2 + v2^2)
    cs = v1 / t
    sn = v2 / t
    return (cs, sn)
end


#%---------------------------------------------------------------------%
#%                  GMRES with two QR                                  %
#%---------------------------------------------------------------------%
function QR_GMRES(A::Matrix, b::Matrix)
    Q = b / norm(b)
    R = Array{Float64}(undef, 1, 1)
    R[1, 1] = norm(b)
    v = A * Q
    tQ = v / norm(v)
    tR = Array{Float64}(undef, 1, 1)
    tR[1, 1] = norm(v)
    ts = size(A, 1)
    zk = zeros(ts, 1)
    Zs = zk
    k = 1
    r = A * zk - b                                   #residual
    r_norm = norm(r, 2)
    while k <= ts && r_norm > 1e-6
        (Q, R) = Arnorldi_QR(Q, R, v[:])
        v = A * Q[:, end]
        (tQ, tR) = Arnorldi_QR(tQ, tR, v[:])
        α = tR \ (tQ' * b)
        zk = Q * α
        Zs = hcat(Zs, zk)
        r = A * zk - b
        r_norm = norm(r, 2)
        k = k + 1
    end
    return zk, Zs
end


#%----------------------------------------------------%
#%                 QR with Arnoldi
#%----------------------------------------------------%
function Arnorldi_QR(Q::Matrix, R::Matrix, x::Vector)
                                                    #compute QR factorization of [QR | x ]:
    (m, n) = size(R)                                #R is supposed to be tall & skinny
    v = x - Q * (Q' * x)
    Q = [Q v / norm(v)]                             #adding an extra column to Q
    R = [[R; zeros(1, n)] Q' * x]                   #adding an extra row (zeros) and column (Q'x) to R
    for k = m:-1:n+1
        p = k:k+1
        (G, R[p, n+1]) = Givens_Rotation(R[p, n+1]) #row k and k+1 of column n+1 of R, a 2-elemente vector.
        if (k < n)
            R[p, k+1:n] = G * R[p, k+1:n]
        end
        Q[:, p] = Q[:, p] * G'
    end
    return (Q, R)
end

## GMRES
function GMRES_Teran(A::Matrix, b::Matrix)            # z0 is 0.
    bnorm = norm(b, 2)
    Q = b / bnorm
    Hk = zeros(1, 0)
    ts = size(A, 1)
    zk = zeros(ts, 1)
    λ = zeros(ts, 1)
    b = zeros(ts, 1)
    b[1, 1] = bnorm
    sn = []; cs = []; G = []                           # Givens Rotation sin&cos
    Zs = []

    for k = 1:ts
        (hk, qk1) = Arnoldi_Teran(A, Q, k)
        Q = [Q qk1]

        (hk, csk, snk) = Compute_Givens_Rotation(hk, cs, sn, k)
        cs = [cs; csk]
        sn = [sn; snk]
        Hk = [[Hk; zeros(1, k - 1)] hk]
        b[k+1] = -sn[k] * b[k]
        b[k] = cs[k] * b[k]

        println("Hk is supposed to be upper-triangular at this point:")
        λ = backsubstitution(Hk[1:k, 1:k], b[1:k])     # solvign the least squre with qr decomposition
        zk = Q * λ
        Zs = hcat(Zs, zk)
    end
    return zk, Zs
end


function Arnoldi_Teran(A::Matrix, Q::Matrix, k::Number)
    qk1 = A * Q[:, end]
    hk = zeros(k + 1, 1)
    for i = 1:k
        hk[i, 1] = Q[:, i]' * qk1
        qk1 = qk1 - hk[i, 1] * Q[:, i]
    end
    hk[k+1, 1] = norm(qk1, 2)                         # the residual
    qk1 = qk1 / hk[k+1, 1]
    return (hk, qk1)
end

#%----------------------------------------------------%
#%                 Given's Rotation
#%----------------------------------------------------%
function Compute_Givens_Rotation(h, cs, sn, k)        # Given's rotation that rotates [a;b] to [r;0]:
    for i = 1:k-1                                     # Apply for ith row
        temp = cs[i] * h[i] + sn[i] * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i] = temp
    end

    v1 = h[k]
    v2 = h[k+1]
    t = sqrt(v1^2 + v2^2)
    cs_k = v1 / t
    sn_k = v2 / t
    h[k] = cs_k * h[k] + sn_k * h[k+1]                # Eliminate H(i + 1, i)
    h[k+1] = 0.0
    return (h, cs_k, sn_k)
end
