import numpy as np
import opt_einsum as oe


def to_integer(weights, bitwidth, normalize=True):
    """Convert weights and biases to integers.

    :param np.ndarray weights: 2D or 4D weight tensor.
    :param np.ndarray biases: 1D bias vector.
    :param int bitwidth: Number of bits for integer conversion.
    :param bool normalize: Whether to normalize weights and biases by the
        common maximum before quantizing.

    :return: The quantized weights and biases.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    max_val = np.max(np.abs(weights)) \
        if normalize else 1
    a_min = -2**bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / max_val * a_max, a_min, a_max).astype(int)
    return weights



### Network class for the SNN
class Network(object):

    def __init__(self, dfa, dropr, evt, norm, rel, delt, dr, init, clp, lim, inputs, hiddens, outputs, threshold_h, threshold_o, T=100, bias=0.0, lr=0.0001, scale=1.0, twin=100, epsilon=2):
        
        print("tell me why")
        self.T = T              # time window
        self.hiddens = hiddens  # number of neurons in hidden layers
        self.outputs = outputs  # number of neurons in the output layer
        self.inputs = inputs    # number of neurons in the input layer
        self.t_h = threshold_h  # threshold ratio for the hidden layers
        self.t_o = threshold_o  # threshold ratio for the output layers
        self.bias = bias        # bias (in terms of percentage of the threshold)
        self.lr = lr            # learning rate
        self.scale = scale      # weight of the direction connection from error network to the forward network
        self.twin = twin        # weight update period = 2*twin ## for 1 update per time window T, twin = T/2-1
        self.epsilon = epsilon  # relaxation period for spikes for the output layer
        self.h = hiddens
        self.evt = evt          # evt=0 -> normal EMSTDP with integer multiplication (pre-synaptic spike count = pre-synaptic spike activity)
                                # evt=1 -> EMSTDP with integer addition (pre-synaptic spike count = 1 for presynaptic spike activity > 1 else =0)
        self.dropr = dropr      # drop rate
        self.dfa = dfa          # feedback (error) network weight mode
                                # dfa = 0 : symmetric weight
                                # dfa = 1 : feedback alignment
                                # dfa = 2 : direct feedback alignment

        if clp != 0:            # clp = 0 : no weight clipping, clp = 1: weight clipping
            self.clp = True
        else:
            self.clp = False

        self.lim = lim          # weight clip limit

        if dr != 0:             # dr = 0 : derivative always = 1 (straight-through estimator), dr = 1 : derivative approximated from shifterd ReLU
            self.dr = True
        else:
            self.dr = False

        self.init = init        # initialization method
                                    # init = 0 : variance scaling
                                    # init = 1 : 2 bit initialization [-sqrt(3/fan-in), 0, +sqrt(3/fan-in)]
                                    # init = 2 : glorot normal
        self.delt = delt        # loss implementation method, combination of ReLU and non-ReLU, delt = 5 default

        self.rel = rel          # rel = 1 : error network with ReLU neurons

        self.norm = norm        # if norm != 0, threshold initialization depends on the value of norm, else threshold init depends on threshold_h and threshold_o
        # if norm == 0.0:
        #     self.clp = True

        self.fac = 100


        self.w_h, self.w_o = self.Init_ForwardWgt(inputs, outputs, hiddens, init)

        self.e_h, self.e_o = self.Init_FeedbackWgt(inputs, outputs, hiddens, dfa)

        self. threshold_h, self.threshold_o, self.ethreshold_h, self.ethreshold_o = self.Init_Threshold(inputs, outputs, hiddens, threshold_h, threshold_o, init, dfa, norm)



        self.lm_h = []          # variable for learning rate momentum
        for k in range(len(self.w_h)):
            self.lm_h.append(np.ones(np.shape(self.w_h[k])))
        self.lm_o = np.ones(np.shape(self.w_o))

        self.energy = 0.0
        self.iterations = 0

        
        # self.w_h = np.transpose(self.w_h)
        # self.w_o = np.transpose(self.w_o)
        # self.w_h_fixed = self.w_o
        # self.w_h[0] = to_integer(self.w_h[0],8)
        # self.w_o = to_integer(self.w_o,8)

        # self.w_h[0] = self.w_h[0].astype(int)
        # self.w_o = self.w_o.astype(int)

        # self.w_h_fixed = self.w_o
        # #transform threshold to integer
        # self.threshold_h[0] = self.threshold_h[0]*self.fac
        # self.threshold_o = self.threshold_o*self.fac

        # self.threshold_h[0] = self.threshold_h[0].astype(int)
        # self.threshold_o = self.threshold_o.astype(int)


    def hook(self,):
        pass


    ### initialize thresholds for the forward and feedback network. They are initialized per layer based on the EMSTDP paper
    def Init_Threshold(self, inputs, outputs, h, threshold_h, threshold_o, init, dfa, norm):
        hiddenThr1 = threshold_h
        outputThr1 = threshold_o
        threshold_h = []
        # hThr1 = inputs*0.1
        hThr = inputs * np.sqrt(3.0 / float(inputs)) / (2.0)
        if init == 2:
            if len(h) > 1:
                hThr = inputs * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
            else:
                hThr = inputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

        threshold_h.append(hThr * hiddenThr1 / float(len(h) + 1))
        for i in range(len(h) - 1):
            if init != 2:
                threshold_h.append(h[i] * (np.sqrt(3.0 / h[i]) / 2.0) * hiddenThr1 / (len(h) - i - 0))
            elif init == 2:
                if (i + 2) < len(h):
                    threshold_h.append(
                        h[i] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (len(h) - i - 1))
                else:
                    threshold_h.append(
                        h[i] * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (len(h) - i - 1))

        if init != 2:
            threshold_o = h[-1] * (np.sqrt(3.0 / h[-1]) / 2.0) * outputThr1

        elif init == 2:
            threshold_o = h[-1] * (np.sqrt(6.0 / h[-1]) / 2.0) * outputThr1

        ethreshold_h = []
        ethreshold_o = []
        if init == 2:
            if len(h) > 1:
                hThr = h[1] * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
            else:
                hThr = outputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

            if norm == 0:
                ethreshold_h.append(hThr * hiddenThr1 / (len(h)))
            else:
                ethreshold_h.append(norm / (len(h)))

            for i in range(len(h) - 1):
                if (i + 2) < len(h):
                    if norm == 0:
                        ethreshold_h.append(
                            h[i + 2] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (i + 2))
                    else:
                        ethreshold_h.append(norm / (i + 2))
                else:
                    if norm == 0:
                        ethreshold_h.append(
                            outputs * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (i + 2))
                    else:
                        ethreshold_h.append(norm / (i + 2))

        if init != 2:
            ts = 1.0
            ehiddenThr1 = hiddenThr1
            eoutputThr1 = outputThr1
            for i in range(len(h)):
                if norm == 0:
                    if i == len(h) - 1:

                        if dfa == 1:
                            ethreshold_h.append(
                                outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                        elif dfa == 2:
                            ethreshold_h.append(
                                outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                        else:
                            ethreshold_h.append(
                                outputs * (np.sqrt(3.0 / h[-1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                    else:
                        if dfa == 1:
                            ethreshold_h.append(
                                h[i + 1] * (np.sqrt(3.0 / h[i + 1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                        elif dfa == 2:
                            ethreshold_h.append(
                                outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                        else:
                            ethreshold_h.append(
                                h[i + 1] * (np.sqrt(3.0 / h[i]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                else:
                    ethreshold_h.append(norm / (1))
        if init == 3:
            tss = 1.0
            ethreshold_h = np.divide(threshold_h, tss)
            ethreshold_o = np.divide(threshold_o, tss)

        return threshold_h, threshold_o, ethreshold_h, ethreshold_o

    ## initialize the feed forward network's weight
    def Init_ForwardWgt(self, inputs, outputs, h, init):
        w_h = []
        w_o = []
        tmpp = np.random.normal(0, np.sqrt(3.0 / float(inputs)), [inputs, h[0]])
        if self.init == 1:
            cut = np.sqrt(3.0 / float(inputs)) * init
            tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
            tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(inputs))
            tmpp[tmpp > cut] = np.sqrt(3.0 / float(inputs))
        elif init == 2:
            if len(h) > 1:
                tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + h[1])), [inputs, h[0]])
            else:
                tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + outputs)), [inputs, h[0]])
        w_h.append(tmpp)
        for i in range(0, len(h) - 1):
            tmpp = np.random.normal(0, np.sqrt(3.0 / h[i]), [h[i], h[i + 1]])
            if init == 1:
                cut = np.sqrt(3.0 / float(h[i])) * init
                tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
                tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[i]))
                tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[i]))
            elif init == 2:
                if (i + 2) < len(h):
                    tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + h[i + 2])), [h[i], h[i + 1]])
                else:
                    tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + outputs)), [h[i], h[i + 1]])
            w_h.append(tmpp)
        
        tmpp = np.random.normal(0, np.sqrt(3.0 / h[-1]), [h[-1], outputs])
        if init == 1:
            cut = np.sqrt(3.0 / float(h[-1])) * init
            tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
            tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[-1]))
            tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[-1]))
        elif init == 2:
            tmpp = np.random.normal(0, np.sqrt(6.0 / h[-1]), [h[-1], outputs])
        w_o = tmpp
        return w_h, w_o


    ## initialize feedback (error) network's weight
    def Init_FeedbackWgt(self, inputs, outputs, h, dfa):
        e_h = []
        e_o = []
        if dfa == 1:
            # tmpp = np.random.normal(0, np.sqrt(3.0 / float(h[0])),
            #                          size=[inputs, h[0]])
            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(h[0])), high=np.sqrt(1.0 / float(h[0])), size=[inputs, h[0]])
            e_h.append(tmpp)
            for i in range(0, len(h) - 1):
                # tmpp = np.random.normal(0, np.sqrt(3.0 / h[i+1]), [h[i], h[i + 1]])

                tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(h[i+1])), high=np.sqrt(1.0 / float(h[i+1])),
                                         size=[h[i], h[i + 1]])
                e_h.append(tmpp)

            # tmpp = np.random.normal(0, np.sqrt(3.0 /float(outputs)), [h[-1], outputs])
            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                     size=[h[-1], outputs])

            e_o = tmpp

        elif dfa == 2:
            # tmpp = np.random.normal(0, np.sqrt(1.0 / float(outputs)),
            #                          size=[inputs, outputs])
            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                     size=[inputs, outputs])

            e_h.append(tmpp)
            for i in range(0, len(h) - 1):
                # tmpp = np.random.normal(0, np.sqrt(3.0 / float(outputs)), [h[i], outputs])

                tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                         size=[h[i], outputs])
                e_h.append(tmpp)

            # tmpp = np.random.normal(0, np.sqrt(3.0 /float(outputs)), [h[-1], outputs])
            tmpp = np.random.uniform(low=-np.sqrt(1.0 / float(outputs)), high=np.sqrt(1.0 / float(outputs)),
                                     size=[h[-1], outputs])

            # tmpp = np.random.rand(h[-1],outputs)/float(outputs.0)
            e_o = tmpp
            
        return e_h, e_o


    ## Training function per batch
    def Train(self, input_spikes, label, bs):


        self.eps = 0.00000001

        # for feed forward network
        n_hidden = len(self.hiddens)

   
        # hidden_spikes = np.zeros()
        hidden_spikes = [1] * n_hidden
        U_h = [1] * n_hidden
        droph =[1] * n_hidden
        droprate = self.dropr
        for i in range(n_hidden):
            hidden_spikes[i] = np.zeros([self.twin, bs, self.hiddens[i]], dtype=bool)
            droph[i] = np.random.rand(self.hiddens[i]) < droprate
        # hidden_spikes = np.zeros([hiddens,T])
        output_spikes = np.zeros([self.twin, bs, self.outputs], dtype=bool)
        for i in range(n_hidden):
            U_h[i] = np.zeros([self.twin, bs, self.hiddens[i]])
        U_o = np.zeros([self.twin, bs, self.outputs])
        self.energy = 0.0


        # for feedback (error) network
        delta = np.zeros([self.twin, bs, self.outputs])
        delta_h = [1] * n_hidden
        for i in range(n_hidden):
            delta_h[i] = np.zeros([self.twin, bs, self.hiddens[i]])

        sdelta = np.zeros([self.twin, bs, self.outputs])
        sdelta_h = [1] * n_hidden
        for i in range(n_hidden):
            sdelta_h[i] = np.zeros([self.twin, bs, self.hiddens[i]])

        """
        
        """



        ## main training loop (main algorithm)
        for t in range(1, self.twin):
            # first phase
            U_h[0][t, :, :] = oe.contract("Bi,ij->Bj", input_spikes[t - 1, :, :], self.w_h[0])  + U_h[0][t - 1, :, :]
            hidden_spikes[0][t, :, :] = U_h[0][t, :, :] >= self.threshold_h[0]
            U_h[0][t, hidden_spikes[0][t, :, :]] = 0
            U_h[0][t, :, droph[0]] = 0

            '''
            correct here
            '''
            # U_h[0] = U_h[0].astype(int)

         
            U_o[t, :, :] = oe.contract("Bi,ij->Bj", hidden_spikes[n_hidden - 1][t - 1, :, :], self.w_o) + U_o[t - 1, :, :] 

            output_spikes[t, :, :] = U_o[t, :, :] >= self.threshold_o
            U_o[t, output_spikes[t, :, :]] = 0

            for k in range(bs):
                delta[t, k, :] = delta[t-1, k, :]
                delta[t, k, label[k]] = delta[t - 1, k, label[k]] + (np.sum(output_spikes[t - 1, k, label[k]], axis=0) < 1).astype(float)
                delta[t, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))] = delta[t-1, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))] -(np.sum(output_spikes[t - 1:t, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))], axis=0) >= 1).astype(float)


                # generating spikes for the loss (error spikes at output layer)
                if self.rel == 0:
                    self.ethreshold_o = self.epsilon
                    sdelta[t, delta[t, :, :] >= self.ethreshold_o] = 1
                    delta[t, delta[t, :, :] >= self.ethreshold_o] = 0
                    sdelta[t, delta[t, :, :] <= -self.ethreshold_o] = -1
                    delta[t, delta[t, :, :] <= -self.ethreshold_o] = 0
                




            delta_h[n_hidden - 1][t, :, :] = oe.contract("Bj,ij->Bi", sdelta[t-1, :, :], self.w_o) + delta_h[n_hidden - 1][t - 1, :, :]
            delta_h[n_hidden - 1][t, :, droph[n_hidden - 1]] = 0

            sdelta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] >= self.ethreshold_h[n_hidden - 1]] = 1.0
            delta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] >= self.ethreshold_h[n_hidden - 1]] = 0.0
            sdelta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] <= -self.ethreshold_h[n_hidden - 1]] = -1.0
            delta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] <= -self.ethreshold_h[n_hidden - 1]] = 0.0




        tmp0 = (np.sum(hidden_spikes[n_hidden - 1][: self.twin, :, :],
                                axis=0, keepdims=True))
        tmp1 = (1*np.sum((sdelta), axis=0, keepdims=True))
        tpp = np.mean(oe.contract("iBj,iBk->Bjk", tmp0, tmp1), axis=0) / float(self.twin)
        newlr = self.lr

        self.w_o +=  (np.multiply(tpp, newlr))

        tmp0 = (np.sum(input_spikes[ :self.twin, :, :], axis=0,
                                keepdims=True))

        tmp1 = (np.sum((sdelta_h[0][:self.twin,:,:]), axis=0, keepdims=True) )



        tpp = np.mean(oe.contract("iBj,iBk->Bjk", tmp0, tmp1), axis=0) / float(self.twin)
        '''
        change here
        '''
        # self.w_h[0] += (self.fac*np.multiply(tpp, newlr)).astype(int)
        self.w_h[0] += np.multiply(tpp, newlr)



        return np.argmax(np.sum(output_spikes[range(self.twin), :,:], axis=0), axis=-1), np.power(self.energy / (self.T - 1), 2)


    # @numba.jit
    # @profile
    def Test(self, input_spikes, bs):
        n_hidden = len(self.hiddens)
        # hidden_spikes = np.zeros()
        hidden_spikes = [1] * n_hidden
        U_h = [1] * n_hidden
        for i in range(n_hidden):
            hidden_spikes[i] = np.zeros([self.twin, bs, self.hiddens[i]], dtype=bool)
        # hidden_spikes = np.zeros([hiddens,T])
        output_spikes = np.zeros([self.twin, bs, self.outputs], dtype=bool)
        for i in range(n_hidden):
            U_h[i] = np.zeros([self.twin, bs, self.hiddens[i]])
        U_o = np.zeros([self.twin, bs, self.outputs])
        # inputs = np.shape(input_spikes)[1]
        self.energy = 0.0

        for t in range(1, self.twin):

            U_h[0][t, :, :] = oe.contract("Bi,ij->Bj", input_spikes[t - 1, :, :], self.w_h[0]) + U_h[0][t - 1, :,
                                                                                                     :] + \
                                  self.threshold_h[0] * self.bias
            hidden_spikes[0][t, :, :] = U_h[0][t, :, :] >= self.threshold_h[0]
            U_h[0][t, hidden_spikes[0][t, :, :]] = 0

            for h in range(1, n_hidden):
                U_h[h][t, :, :] = oe.contract("Bi,ij->Bj", hidden_spikes[h - 1][t - 1, :, :], self.w_h[h]) + U_h[h][
                                                                                                                 t - 1,
                                                                                                                 :, :] + \
                                      self.threshold_h[h] * self.bias
                hidden_spikes[h][t, :, :] = U_h[h][t, :, :] >= self.threshold_h[h]
                U_h[h][t, hidden_spikes[h][t, :, :]] = 0

            U_o[t, :, :] = oe.contract("Bi,ij->Bj", hidden_spikes[n_hidden - 1][t - 1, :, :], self.w_o) + U_o[t - 1,
                                                                                                              :,
                                                                                                              :] + self.threshold_o * self.bias
            output_spikes[t, :, :] = U_o[t, :, :] >= self.threshold_o
            U_o[t, output_spikes[t, :, :]] = 0
        
        prepared_data = np.sum(hidden_spikes[0][:,:,:], axis=0)/self.twin
        return np.argmax(np.sum(output_spikes[:,:,:], axis=0), axis=-1),prepared_data


#
# # plotting the spikes while debugging
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.imshow(output_spikes[:,0,:].T)
# plt.title('output layer')
# for i in range(n_hidden):
#     plt.figure()
#     plt.imshow(hidden_spikes[i][:,0,:].T)
#     plt.title('layer '+str(i))
#
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.imshow(sdelta[:,0,:].T)
# plt.title('output layer delta')
# for i in range(n_hidden):
#     plt.figure()
#     plt.imshow(sdelta_h[i][:,0,:].T)
#     plt.title('layer delta before derivative'+str(i))
# #     # plt.figure()
# #     # plt.imshow(dh[i])
# #     # plt.title('layer delta after derivative' + str(i))
#
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.imshow(delta[:,0,:].T)
# plt.title('output layer delta')
# plt.figure()
# plt.hist(self.w_o)
# plt.title('output layer hist')
# for i in range(n_hidden):
#     plt.figure()
#     plt.imshow(delta_h[i][:,0,:].T)
#     plt.colorbar()
#     plt.title('layer delta before derivative'+str(i))
#     plt.figure()
#     plt.hist(self.w_h[i])
#     plt.title('layer hist' + str(i))

#
# # import matplotlib.pyplot as plt
# # tmp0 = (np.sum(input_spikes[t - self.twin + 1:t + 1, :, :], axis=0, keepdims=True))
# # tmp1 = (np.sum(hidden_spikes[0][t - self.twin + 1:t + 1, :, :], axis=0, keepdims=True) - np.sum(
# #                         hidden_spikes[0][t - self.twin + 1 - self.twin:t + 1 - self.twin, :, :], axis=0, keepdims=True))
# # a = np.mean(np.array(map(lambda mm: np.dot(tmp0[:, mm, :].T, tmp1[:, mm, :]), range(bs))),
# #                                         axis=0) * ((self.lr * bs * np.sqrt(3.0 / 784) / 2.0) / (self.twin * self.twin))
# #
# # plt.hist(a)