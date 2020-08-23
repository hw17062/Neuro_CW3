#!/usr/bin/python

import sys, getopt
import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import math
from abc import ABC, abstractmethod
from statistics import mean
import time
import gc

mpl.rcParams['agg.path.chunksize'] = 10000

class baseNeuron(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def add_SynapseIn(self, synapse):
        pass

    @abstractmethod
    def add_SynapseOut(self, synapse):
        pass


class LIFNeuron(baseNeuron):
    def __init__(self, tau,volt, volt_reset, leaky, volt_Thesh, **kwargs):
        self.tau = tau * mill
        self.volt = volt * mill
        self.volt_reset = volt_reset * mill
        self.volt_rest = self.volt_reset
        self.leaky = leaky * mill
        self.volt_Thesh = volt_Thesh * mill
        self.Rm_Ie = None
        self.Rm = None
        self.Ie = None
        if ("RmIe" in kwargs):
            self.Rm_Ie = kwargs["RmIe"] * mill
        else:
            self.Rm = kwargs["Rm"] * mega
            self.Ie = kwargs["Ie"] * nano
            self.Rm_Ie = self.Rm * self.Ie

        self.voltage_History= [self.volt]   #Create array with first ele being the starting Voltage

        self.last_Spike_t = 0

        self.spike_count = 0

        self.synapseInList = []
        self.synapseOutList = []


    # Performill one timestep update to the LIFNeuron and saves the voltage in the history
    def update(self):
        synapse_input = self.calc_syn_input()
        # print(synapse_input)

        if self.volt >= self.volt_Thesh:   # If we spike, reset
            self.volt = self.volt_reset
            self.last_Spike_t = currentTime
            self.spike_count += 1
            self.spike_Synapses()

        # dV = ((El - V) + RmIe)*dt/tau
        diff = ((self.leaky - self.volt) + (self.Rm_Ie + synapse_input)) *  (timestep/self.tau)
        self.volt += diff

        if (self.volt < self.volt_rest): self.volt = self.volt_rest

        self.voltage_History.append(self.volt)

    def add_SynapseIn(self, synapse):
        self.synapseInList.append(synapse)

    def add_SynapseOut(self, synapse):
        self.synapseOutList.append(synapse)

    def calc_syn_input(self):
        input = 0
        for synapse in self.synapseInList:
            input += synapse.RmIs

        return input

    def spike_Synapses(self):
        for synapse in self.synapseOutList:
            synapse.spike()

        if (self.synapseInList):
            if self.synapseInList[0].__class__.__name__ == "STDPSynapse":
                for synapse in self.synapseInList:
                    synapse.STDP_Update()

    def reset_Spike_Count(self):
        self.spike_count = 0

class BaseSynapse(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def spike(self):
        pass

class Synapse(BaseSynapse):
    def __init__(self, pre_LIFNeuron, post_LIFNeuron, RmG,Delta_s, tauS, Es):
        self.pre_LIFNeuron = pre_LIFNeuron
        self.post_LIFNeuron = post_LIFNeuron
        self.RmGs = RmG
        self.tauS = tauS * mill
        self.Es = Es * mill
        self.delta_s = Delta_s

        self.s = 0

        self.RmIs = 0
        self.pre_LIFNeuron.add_SynapseOut(self)

        self.post_LIFNeuron.add_SynapseIn(self)

    def update(self):
        self.s = self.s - (self.s * timestep) /self.tauS
        self.RmIs = self.RmGs * (self.Es - self.post_LIFNeuron.volt) * self.s

    def spike(self):
        self.s += self.delta_s

"This is a class to to create the neurons needed in part 2."
"It will simulate a poisson spike train to decide on a 'spike'"
class poissonSynapse(BaseSynapse):
    def __init__(self, post_LIFNeuron, Gi, Delta_s, tauS, Es, Firerate):
        self.post_LIFNeuron = post_LIFNeuron
        self.Gi = Gi * nano
        self.tauS = tauS * mill
        self.Es = Es    * mill
        self.delta_s = Delta_s
        self.Firerate = Firerate
        self.Rm = self.post_LIFNeuron.Rm
        self.RmGs = self.Rm*self.Gi

        self.s = 0
        self.RmIs = 0

        self.last_spike_recieved_t = 0

        self.post_LIFNeuron.add_SynapseIn(self)
    def update(self):
        # Now we check for a spike from the poisson 'Neuron'
        chance = rnd.random()


        diff = -self.s/self.tauS * timestep
        self.s += diff

        if (chance < timestep*self.Firerate):
            self.spike()
            self.last_spike_recieved_t = currentTime
        # self.s = self.s - (self.s * timestep) / self.tauS
        self.RmIs = self.RmGs * (self.Es - self.post_LIFNeuron.volt) * self.s


    def spike(self):
        self.s += self.delta_s

class STDPSynapse(poissonSynapse):
    def __init__(self, aPlus, aMinus, tauPlus, tauMinus, sV):

        self.aPlus = aPlus * nano
        self.aMinus = aMinus * nano
        self.tauPlus = tauPlus * mill
        self.tauMinus = tauMinus * mill

        self.last_spike_recieved_t = 0

        poissonSynapse.__init__(self, sV[0], sV[1], sV[2], sV[3], sV[4], sV[5])

        self.Gi_history = [self.Gi]

    #Slightly updated spike to now recordthe spike time and change according to STDP
    def spike(self):
        self.s += self.delta_s
        self.last_spike_recieved_t = currentTime
        self.STDP_Update()

    # Calculate the spike time differance and update the Gi accordingly
    def STDP_Update(self):
        delta_t =  self.post_LIFNeuron.last_Spike_t - self.last_spike_recieved_t
        if delta_t > 0:
            self.Gi = self.Gi + (self.aPlus * math.exp(-abs(delta_t)/self.tauPlus))
        else:
            self.Gi = self.Gi + (-self.aMinus * math.exp(-abs(delta_t)/self.tauMinus))

        # Limit Gi in case it gets too large or negative from update
        if self.Gi > 4 * nano:
            self.Gi = 4 * nano
        elif self.Gi < 0:
            self.Gi = 0

        self.Gi_history.append(self.Gi)

        self.RmGs = self.Rm * self.Gi

def runSimulation(Neurons, Synapses):
    global currentTime
    for t in timestamps:
        currentTime = t
        for syn in Synapses:
            syn.update()
        for n in Neurons:
            n.update()


    return Neurons, Synapses

# -----------------------------------------------------------------------------
#                           Question 1

# Over 1 second with the following paramerters:
# TAUm = 10mill,  El = Vrest = -70mV, Vth = -40mV
# R, = 10M, Ie = 3.1nA
# Delta_t = 0.25


# Cm . dV/dt = (El - V)/Rm + Ie
# dV = (((El - V)/Rm) + Ie)/Cm * dt

# dV = ((El - V) + RmIe)*dt/tau

# V = voltage
# I = current
# R = Resistance (membrane)
# El = Leak potential
# TAUm = membrane time constant

# V thresh = threshhold of voltage for a 'spike'
# V reset  = volt value after a spike, what it resets to


def QuestionOne():
    #                    (tau,volt, volt_reset, leaky, volt_Thesh, Rm_Ie):
    lifNeuron = LIFNeuron(10 ,-70 ,        -70,   -70,        -54,   Rm=8, Ie=2.1)

    runSimulation([lifNeuron], [])

    #for i in range(len(timestamps) - 1):    # Loop through how many time steps there are
    #    lifNeuron.update()

    fig, ax = plt.subplots(1,1)

    ax.plot(lifNeuron.voltage_History)

    # Now we set up the plot, firstly we get the xTicks
    lables = np.linspace(0,1, num=5)
    xLoc = np.linspace(0,len(lifNeuron.voltage_History), num=5)
    plt.xticks(xLoc,lables)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("LIF Model")

    print("Saving figure_p1_1_resit.png after - %s seconds -" % (time.time() - start_time))

    plt.savefig('figure_p1_1_resit.png', bbox_inches='tight', dpi=500)


# -----------------------------------------------------------------------------
#                           Question 2

# Class to contain everything needed for an LIF LIFNeuron model

def QuestionTwo(Es):

    #                     (tau,volt, volt_reset, leaky, volt_Thesh, Rm_Ie):
    volt = rnd.randrange(-80, -54)
    lifNeuron1 = LIFNeuron(20,volt ,-80        ,-70   ,-54        ,RmIe=18)
    volt = rnd.randrange(-80, -54)
    lifNeuron2 = LIFNeuron(20  ,volt ,-80        ,-70   ,-54        ,RmIe=18)

    #                   (pre_LIFNeuron, post_LIFNeuron, RmG  ,Delta_s, tauS, Es)
    synapse1_2 = Synapse(lifNeuron1   , lifNeuron2    , 0.15 , 0.5   , 10 , Es)
    synapse2_1 = Synapse(lifNeuron2   , lifNeuron1    , 0.15 , 0.5   , 10 , Es)

    Ns = [lifNeuron1, lifNeuron2]
    Ss = [synapse1_2, synapse2_1]
    Ns, Ss = runSimulation(Ns, Ss)

    # for i in range(len(timestamps) - 1):    # Loop through how many time steps there are
    #     lifNeuron1.update()
    #     lifNeuron2.update()
    #     synapse1_2.update()
    #     synapse2_1.update()


    fig, ax = plt.subplots(1,1)

    ax.plot(Ns[0].voltage_History, color = 'blue')
    ax.plot(Ns[1].voltage_History, color = 'red')
    ax.legend(['lifNeuron One', 'lifNeuron Two'])

    # Now we set up the plot, firstly we get the xTicks
    lables = np.linspace(0,1, num=5)
    xLoc = np.linspace(0,len(Ns[0].voltage_History), num=5)
    plt.xticks(xLoc,lables)
    # plt.xticks(np.arange(0,len(Ns[0].voltage_History), step=len(Ns[0].voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("LIF Model, Es = {}".format(Es))

    part = "a" if Es == 0 else "b"

    print("Saving figure_p1_2{}_resit.png after - {} seconds -".format(part, time.time() - start_time))
    plt.savefig('figure_p1_2{}_resit.png'.format(part), bbox_inches='tight', dpi=500)

# =============================================================================
#                           Part 2


# -----------------------------------------------------------------------------
#                           Question 1

def PartB_QuestionOne():
    #                    (tau,volt, volt_reset, leaky, volt_Thesh, RmIe)
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)
    #                          (post_LIFNeuron, Gi, Delta_s, tauS, Es, Firerate):
    PSynapses = [poissonSynapse(lifNeuron     , 4 ,0.5     , 2   , 0 , 15 ) for i  in range(40)]

    print(lifNeuron.Rm_Ie)
    lifNeuron, PSynapses = runSimulation([lifNeuron], PSynapses)

    fig, ax = plt.subplots()

    ax.plot(lifNeuron[0].voltage_History, color = 'blue')
    ax.hlines(lifNeuron[0].volt_Thesh, 0,4000)

    # Now we set up the plot, firstly we get the xTicks
    lables = np.linspace(0,1, num=5)
    xLoc = np.linspace(0,len(lifNeuron[0].voltage_History), num=5)
    plt.xticks(xLoc,lables)
    # plt.xticks(np.arange(0,len(lifNeuron[0].voltage_History), step=len(lifNeuron[0].voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("Part 2 Q1")

    print("Saving figure_p2_1_resit.png after - %s seconds -" % (time.time() - start_time))
    plt.savefig('figure_p2_1_resit.png', bbox_inches='tight', dpi=500)


# -----------------------------------------------------------------------------
#                           Question 2

def PartB_QuestionTwo(stdp):
    global currentTime
    #                    (tau,volt, volt_reset, leaky, volt_Thesh, RmIe)
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)

    synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 15]
    #                  (post_LIFNeuron, Gi       ,Delta_s , tauS, Es, Firerate):
    if (not stdp):
        PSynapses = [poissonSynapse(lifNeuron     , 4    ,0.5     , 2   , 0 , 15) for i  in range(40)]
    else:
        #                       (aPlus, aMinus, tauPlus, tauMinus, *sV)
        PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(40)]

    spike_Counter_Bins = []

    for t in timestamps:
        currentTime = t
        lifNeuron.update()
        for syn in PSynapses:
            syn.update()


        if (t % 10 < timestep/2 and not t == 0):
            spike_Counter_Bins.append(lifNeuron.spike_count/10)
            lifNeuron.reset_Spike_Count()

    spike_Counter_Bins.append(lifNeuron.spike_count/(currentTime % 10))
    lifNeuron.reset_Spike_Count()

    filename = ""
    final_Gis = []

    # create 2 in one grpasg for Gis and fire rate if useing STDP
    if (stdp):
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

        filename = "figure_p2_2a_resit.png"
        fig.suptitle("Part 2 Q2 w/ stdp")

        for syn in PSynapses:
            final_Gis.append(syn.Gi_history[-1])

        ax1.hist(final_Gis)

        ax1.set_xlabel("Gi Values (nS)")
        ax1.set_ylabel("# Of Synapses")

        ax2.plot(spike_Counter_Bins, color = 'blue')
        #ax.hlines(lifNeuron.volt_Thesh, 0,4000)

        # Now we set up the plot, firstly we get the xTicks
        lables = np.around(np.linspace(0,timestamps[-1], num=11))
        xLoc = np.around(np.linspace(0,len(spike_Counter_Bins), num=11))
        ax2.set_xticks(xLoc)
        ax2.set_xticklabels(lables)
        # plt.xticks(np.arange(0,len(spike_Counter_Bins) + 1, step=len(spike_Counter_Bins)/6),  [str(i * duration/6) for i in range(7)])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Firerate (Hz)")
        fig.tight_layout(pad=1.0)
    else:
        fig, ax = plt.subplots()

        filename = "figure_p2_2b_resit.png"
        fig.suptitle("Part 2 Q2 w/o stdp")

        ax.plot(spike_Counter_Bins, color = 'blue')
        #ax.hlines(lifNeuron.volt_Thesh, 0,4000)

        # Now we set up the plot, firstly we get the xTicks
        lables = np.around(np.linspace(0,timestamps[-1], num=11))
        xLoc = np.around(np.linspace(0,len(spike_Counter_Bins), num=11))
        ax.set_xticks(xLoc)
        ax.set_xticklabels(lables)
        # plt.xticks(np.arange(0,len(spike_Counter_Bins) + 1, step=len(spike_Counter_Bins)/6),  [str(i * duration/6) for i in range(7)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firerate (Hz)")

    print("Saving {} after - {} seconds -".format(filename, (time.time() - start_time)))
    fig.savefig(filename, bbox_inches='tight')


# -----------------------------------------------------------------------------
#                           Question 3

def PartB_QuestionThree():

    #gets the spike counter from each firerate
    spike_Counter_Bins = []
    spike_Counter_Bins_stdp = []
    stdp = True

    # Loop throught 10-20 hz firerate and one for stdp and non-stdp synapses
    for i in range(2):
        stdp = not stdp
        for j in range(10):
            print("On simulation: ", i*10 + j)
            #                    (tau,volt, volt_reset, leaky, volt_Thesh, RmIe)
            lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)

            synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 10 + j]   #ease of use for vars being sent ot the synapse
            #                  (post_LIFNeuron, Gi   ,Delta_s ,tauS ,Es , Firerate):
            if (not stdp):
                PSynapses = [poissonSynapse(lifNeuron     , 4    ,0.5     , 2   , 0 , 10 + j) for i  in range(40)]
                runSimulation([lifNeuron], PSynapses)   #run the sim
                spike_Counter_Bins.append(lifNeuron.spike_count/300)  # Add the total spike count to a list
            else:
                #                       (aPlus, aMinus, tauPlus, tauMinus, *sV)
                PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(40)]
                runSimulation([lifNeuron], PSynapses)   # Run the sim
                spike_Counter_Bins_stdp.append(lifNeuron.spike_count/300)    # Add the total spike count to a list

            del PSynapses
            gc.collect()

    fig, ax = plt.subplots(1,1)

    ax.plot(spike_Counter_Bins, color = 'blue')
    ax.plot(spike_Counter_Bins_stdp, color = 'red')

    # Now we set up the plot, firstly we get the xTicks
    lables = np.around(np.linspace(10,20, num=11), 0)
    xLoc = np.around(np.linspace(0,len(spike_Counter_Bins_stdp), num=11))
    ax.set_xticks(xLoc)
    ax.set_xticklabels(lables)

    # plt.xticks(np.arange(0,len(spike_Counter_Bins_stdp), step=len(spike_Counter_Bins_stdp)/5),  [str(round(i * len(spike_Counter_Bins_stdp)/6, 3)) for i in range(7)])
    plt.xlabel("input Firerate (hz)")
    plt.ylabel("output firerates")

    plt.title("Part 2 Q3")
    ax.legend(['Non-STDP', 'w/ STDP'])

    #plt.show()

    print("Saving figure_p2_3a_resit.png after - %s seconds -" % (time.time() - start_time))
    plt.savefig('figure_p2_3a_resit.png', bbox_inches='tight', dpi=500)

def PartB_QuestionThree_Hist():


    Gi_vals_mean = []
    for i in range(2):
        print("On simulation: ", i)
        lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)
        synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 10 * (i + 1)]   #ease of use for vars being sent ot the synapse
        PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(40)]

        runSimulation([lifNeuron], PSynapses)

        #gather all the synapses strength in the last 30s of the simulation
        Gi_Vals_Holder = [PSynapses[s].Gi_history[-(int(30/timestep)):] for s in range(40)]

        # now get the means strengths
        Gi_vals_mean.append([mean(Gi_Vals_Holder[j]) for j in range(40)])

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all')
    ax1.hist(Gi_vals_mean[0], color = 'blue')
    ax2.hist(Gi_vals_mean[1], color = 'red')

    # plt.xticks(np.arange(0,len(Gi_vals_mean[1]), step=len(Gi_vals_mean[1])/4),  [str(round(i * len(Gi_vals_mean[1])/5, 3)) for i in range(6)])

    # Now we set up the plot, firstly we get the xTicks
    # lables = np.around(np.linspace(3,4, num=11))
    # xLoc = np.around(np.linspace(3,4, num=11))
    # ax1.set_xticks(xLoc)
    # ax1.set_xticklabels(lables)
    # ax2.set_xticks(xLoc)
    # ax2.set_xticklabels(lables)
    # fig.ylabel("Steady-state synaptic strengths")

    fig.suptitle("Part 2 Q3, STDP <r> = 10 and 20")

    ax1.set_xlabel("Gi Values (nS)")
    ax1.set_ylabel("# Of Synapses")

    ax2.set_xlabel("Gi Values (nS)")
    ax2.set_ylabel("# Of Synapses")

    ax1.xaxis.set_tick_params(which='both', labelbottom=True)

    fig.legend(['10Hz', '20Hz'], loc= (0.2,0.8))
    fig.tight_layout(pad=1.0)

    print("Saving figure_p2_3b_resit.png after - %s seconds -" % (time.time() - start_time))
    plt.savefig('figure_p2_3b_resit.png', bbox_inches='tight', dpi=500)


#  -----------------------------------------------------------------------------
#                           Question 3

def PartB_QuestionFour():
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)
    synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 10]   #ease of use for vars being sent ot the synapse
    PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(20)]
    synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 20]   #ease of use for vars being sent ot the synapse
    PSynapses.extend([STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(20)])

    runSimulation([lifNeuron], PSynapses)

    Gi_Vals_Holder = [PSynapses[s].Gi_history[-(int(30/timestep)):] for s in range(40)]
    Gi_vals_mean = []
    Gi_vals_mean.append([mean(Gi_Vals_Holder[j]) for j in range(20)])
    Gi_vals_mean.append([mean(Gi_Vals_Holder[j + 20]) for j in range(20)])

    print("r1 Mean = {} \nr2 Mean = {}".format(mean(Gi_vals_mean[0]), mean(Gi_vals_mean[1])))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all')
    ax1.hist(Gi_vals_mean[0], color = 'blue')
    ax2.hist(Gi_vals_mean[1], color = 'red')

    fig.suptitle("Part 2 Q4, STDP = on, <r1> = 10, <r2> = 20")

    ax1.set_xlabel("Gi Values (nS)")
    ax1.set_ylabel("# Of Synapses")

    ax2.set_xlabel("Gi Values (nS)")
    ax2.set_ylabel("# Of Synapses")

    ax1.xaxis.set_tick_params(which='both', labelbottom=True)

    fig.legend(['<r1>', '<r2>'], loc= (0.2,0.8))
    fig.tight_layout(pad=1.0)

    print("Saving figure_p2_4a_resit.png after - %s seconds -" % (time.time() - start_time))
    plt.savefig('figure_p2_4a_resit.png', bbox_inches='tight', dpi=500)
    print("Saved figure_p2_4a_resit.png after - %s seconds -" % (time.time() - start_time))

    #r1 Mean = 3.553478363726296e-09
    #r2 Mean = 1.3467810247457746e-09

    # mean = 2.4501296942360353

    #r3 Mean = 2.179196030496824e-09


def PartB_QuestionFour_15():
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)
    synapseVariables = [lifNeuron     , 4    ,0.5     , 2   , 0 , 15]   #ease of use for vars being sent ot the synapse
    PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(40)]

    runSimulation([lifNeuron], PSynapses)

    Gi_Vals_Holder = [PSynapses[s].Gi_history[-(int(30/timestep)):] for s in range(40)]
    Gi_vals_mean = []
    Gi_vals_mean.append([mean(Gi_Vals_Holder[j]) for j in range(40)])

    print("r3 Mean = {}".format(mean(Gi_vals_mean[0])))

    fig, ax1 = plt.subplots()
    ax1.hist(Gi_vals_mean[0], color = 'blue')

    fig.suptitle("Part 2 Q4, STDP = on, <r> = 15")

    ax1.set_xlabel("Gi Values (nS)")
    ax1.set_ylabel("# Of Synapses")
    fig.tight_layout(pad=1.0)

    print("Saving figure_p2_4b_resit.png after - %s seconds -" % (time.time() - start_time))
    plt.savefig('figure_p2_4b_resit.png', bbox_inches='tight', dpi=500)
    print("Saved figure_p2_4b_resit.png after - %s seconds -" % (time.time() - start_time))


# -----------------------------------------------------------------------------
#                           ~~ MAIN ~~

mill = 0.001
mega = 1000000
nano = 0.000000001
timestep = 0.25 * mill

timestamps = []
currentTime = 0
duration = 1
start_time = 0

def main(argv):
    global duration
    global firerate
    global timestamps

    input_duration = duration   # Defult = 1
    global start_time
    start_time = time.time()
    STDP = False

    try:
      opts, args = getopt.getopt(argv,"sd:",["stdp"])
    except getopt.GetoptError:
      print ('main.py [-s, -d]')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-s", "--stdp"):
          STDP = True
      elif opt == '-d':
          input_duration = int(arg)

    # Set durations of the Qs appropriately
    #duration = 1
    #timestamps = np.arange(0, duration, timestep)
    #QuestionOne()

    # duration = 1
    # timestamps = np.arange(0, duration, timestep)
    #QuestionTwo(0)
    #QuestionTwo(-80)

    duration = 300
    timestamps = np.arange(0, duration, timestep)
    # PartB_QuestionOne()
    PartB_QuestionTwo(True)
    PartB_QuestionTwo(False)

    # duration = 300
    # timestamps = np.arange(0, duration, timestep)
    PartB_QuestionThree()
    PartB_QuestionThree_Hist()

    # duration = 300
    # timestamps = np.arange(0, duration, timestep)
    # PartB_QuestionFour()
    # PartB_QuestionFour_15()

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
   main(sys.argv[1:])
