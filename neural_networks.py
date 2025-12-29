import numpy as np

# Define the inputs
i1 = 0.1
i2 = 0.2

# Define input vector
iv = np.array([[i1],[i2]])

# Define the targets
ot1 = 0.05
ot2 = 0.95

# Define the inital weights and biases
w1, w2, w3, w4, w5, w6, w7, w8 = np.ones(8) * 0.2
b1, b2, b3, b4 = np.ones(4) * 0.3

# Initial weight and bias vector
wbv = np.array([w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4])

learning_rate = 0.5
amount_of_loops = 5000
for i in range(amount_of_loops):

    # Define bias vectors
    bh = np.array([[wbv[8]],[wbv[9]]])
    bo = np.array([[wbv[10]],[wbv[11]]])

    # Define the weight matrices
    wh = np.array([[wbv[0], wbv[1]],[wbv[2], wbv[3]]])
    wo = np.array([[wbv[4], wbv[5]],[wbv[6], wbv[7]]])

    # Forward pass
    h = wh @ iv + bh
    hs = 1 / (1 + np.exp(-h))
    o = wo @ hs + bo
    os = 1 / (1 + np.exp(-o))

    h1 = h[0][0]
    h2 = h[1][0]

    hs1 = hs[0][0]
    hs2 = hs[1][0]

    o1 = o[0][0]
    o2 = o[1][0]

    os1 = os[0][0]
    os2 = os[1][0]

    Error_total = 0.5 * ((ot1 - os1)**2 + (ot2 - os2)**2)
    print(f"Error_total: {Error_total}")

    # Backwards pass
    dhs1_dh1 = np.exp(-h1) / (1 + np.exp(-h1))**2
    dhs2_dh2 = np.exp(-h2) / (1 + np.exp(-h2))**2
    dos1_do1 = np.exp(-o1) / (1 + np.exp(-o1))**2
    dos2_do2 = np.exp(-o2) / (1 + np.exp(-o2))**2

    Ew1G = ((os1 - ot1) * dos1_do1 * wbv[4] + (os2 - ot2) * dos2_do2 * wbv[6]) * dhs1_dh1 * i1
    Ew2G = ((os1 - ot1) * dos1_do1 * wbv[4] + (os2 - ot2) * dos2_do2 * wbv[6]) * dhs1_dh1 * i2
    Ew3G = ((os1 - ot1) * dos1_do1 * wbv[5] + (os2 - ot2) * dos2_do2 * wbv[7]) * dhs2_dh2 * i1
    Ew4G = ((os1 - ot1) * dos1_do1 * wbv[5] + (os2 - ot2) * dos2_do2 * wbv[7]) * dhs2_dh2 * i2
    Ew5G = (os1 - ot1) * dos1_do1 * hs1
    Ew6G = (os1 - ot1) * dos1_do1 * hs2
    Ew7G = (os2 - ot2) * dos2_do2 * hs1
    Ew8G = (os2 - ot2) * dos2_do2 * hs2
    Eb1G = ((os1 - ot1) * dos1_do1 * wbv[4] + (os2 - ot2) * dos2_do2 * wbv[6]) * dhs1_dh1 * 1
    Eb2G = ((os1 - ot1) * dos1_do1 * wbv[5] + (os2 - ot2) * dos2_do2 * wbv[7]) * dhs2_dh2 * 1
    Eb3G = (os1 - ot1) * dos1_do1 * 1
    Eb4G = (os2 - ot2) * dos2_do2 * 1

    EG = np.array([Ew1G, Ew2G, Ew3G, Ew4G, Ew5G, Ew6G, Ew7G, Ew8G, Eb1G, Eb2G, Eb3G, Eb4G])

    wbv = wbv - learning_rate * EG


print(f"Final output: {os}")
print(f"Weight vector: {wbv}")

























#############
