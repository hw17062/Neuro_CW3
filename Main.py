#!/usr/bin/python

import sys, getopt
import random as rnd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import math
from abc import ABC, abstractmethod

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

        self.voltage_History = np.array([self.volt], float)    #Create array with first ele being the starting Voltage

        self.last_Spike_t = 0

        self.synapseInList = np.array([])
        self.synapseOutList = np.array([])


    # Performill one timestep update to the LIFNeuron and saves the voltage in the history
    def update(self):
        synapse_input = self.calc_syn_input()

        if self.volt >= self.volt_Thesh:   # If we spike, reset
            self.volt = self.volt_reset
            self.last_Spike_t = currentTime
            self.spike_Synapses()

        # dV = ((El - V) + RmIe)*dt/tau
        diff = ((self.leaky - self.volt) + (self.Rm_Ie + synapse_input)) *  timestep/self.tau
        self.volt += diff

        if (self.volt < self.volt_rest): self.volt = self.volt_rest

        self.voltage_History = np.append(self.voltage_History, [self.volt])

    def add_SynapseIn(self, synapse):
        self.synapseInList = np.append(self.synapseInList, [synapse])

    def add_SynapseOut(self, synapse):
        self.synapseOutList = np.append(self.synapseOutList, [synapse])

    def calc_syn_input(self):
        input = 0
        for synapse in self.synapseInList:
            input += synapse.RmIs
        return input

    def spike_Synapses(self):
        for synapse in self.synapseOutList:
            synapse.spike()

        if self.synapseInList[0].__class__.__name__ == "STDPSynapse":
            for synapse in self.synapseInList:
                synapse.STDP_Update()

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

        self.post_LIFNeuron.add_SynapseIn(self)

    def update(self):
        # Now we check for a spike from the poisson 'Neuron'
        chance = rnd.random()
        if (chance < timestep*self.Firerate):
            self.spike()
            self.last_spike_recieved_t = currentTime

        self.s = self.s - (self.s * timestep) / self.tauS
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

        self.Gi_history = np.array([self.Gi])

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
        elif delta_t <= 0:
            self.Gi = self.Gi + (-self.aMinus * math.exp(-abs(delta_t)/self.tauMinus))

        # Limit Gi in case it gets too large or negative from update
        if self.Gi > 4 * nano:
            self.Gi = 4 * nano
        elif self.Gi < 0:
            self.Gi = 0

        self.Gi_history = np.append(self.Gi_history, self.Gi)



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
    lifNeuron = LIFNeuron(10 ,-70 ,        -70,   -70,        -40,   Rm=10, Ie=3.1)
    for i in range(len(timestamps) - 1):    # Loop through how many time steps there are
        lifNeuron.update()

    fig, ax = plt.subplots(1,1)

    ax.plot(lifNeuron.voltage_History)

    #plt.xticks(np.arange(0,len(LIFNeuron.voltage_History), step=len(LIFNeuron.voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("LIF Model")

    plt.show()


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


    for i in range(len(timestamps) - 1):    # Loop through how many time steps there are
        lifNeuron1.update()
        lifNeuron2.update()
        synapse1_2.update()
        synapse2_1.update()


    fig, ax = plt.subplots(1,1)

    ax.plot(lifNeuron1.voltage_History, color = 'blue')
    ax.plot(lifNeuron2.voltage_History, color = 'red')
    ax.legend(['lifNeuron One', 'lifNeuron 2'])

    plt.xticks(np.arange(0,len(lifNeuron1.voltage_History), step=len(lifNeuron1.voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("LIF Model")

    plt.show()

# =============================================================================
#                           Part 2


# -----------------------------------------------------------------------------
#                           Question 1

def PartB_QuestionOne():
    #                    (tau,volt, volt_reset, leaky, volt_Thesh, RmIe)
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)
    #                          (post_LIFNeuron, Gi, Delta_s, tauS, Es, Firerate):
    PSynapses = [poissonSynapse(lifNeuron     , 4 ,0.5     , 2   , 0 , 15 ) for i  in range(40)]

    for t in timestamps:
        lifNeuron.update()
        for syn in PSynapses:
            syn.update()
        currentTime = t

    fig, ax = plt.subplots(1,1)

    ax.plot(lifNeuron.voltage_History, color = 'blue')
    ax.hlines(lifNeuron.volt_Thesh, 0,4000)

    plt.xticks(np.arange(0,len(lifNeuron.voltage_History), step=len(lifNeuron.voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Value")

    plt.title("Part 2 Q1")

    plt.show()


# -----------------------------------------------------------------------------
#                           Question 2

def PartB_QuestionTwo(stdp):
    global currentTime
    #                    (tau,volt, volt_reset, leaky, volt_Thesh, RmIe)
    lifNeuron = LIFNeuron(10 ,-65 ,-65        ,-65   ,-50        ,Rm =100, Ie = 0)

    synapseVariables = [lifNeuron     , 4 ,0.5     , 2   , 0 , 15]
    #                              (post_LIFNeuron, Gi, Delta_s, tauS, Es, Firerate):
    if (not stdp):
        PSynapses = [poissonSynapse(lifNeuron     , 4 ,0.5     , 2   , 0 , 15 ) for i  in range(40)]
    else:
        #                       (aPlus, aMinus, tauPlus, tauMinus, *sV)
        PSynapses = [STDPSynapse(0.2, 0.25, 20, 20,synapseVariables) for i  in range(40)]

    for t in timestamps:
        lifNeuron.update()
        for syn in PSynapses:
            syn.update()
        currentTime = t


    fig, ax = plt.subplots(1,1)

    final_Gis = np.array([])
    for syn in PSynapses:
        final_Gis = np.append(final_Gis, syn.Gi_history[-1])

    ax.hist(final_Gis)


    #ax.plot(lifNeuron.voltage_History, color = 'blue')
    #ax.hlines(lifNeuron.volt_Thesh, 0,4000)

    #plt.xticks(np.arange(0,len(lifNeuron.voltage_History), step=len(lifNeuron.voltage_History)/5), ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.xlabel("Gi Values (nA)")
    plt.ylabel("# Of Synapses")

    plt.title("Part 2 Q2")

    plt.show()


mill = 0.001
mega = 1000000
nano = 0.000000001
timestep = 0.25 * mill

timestamps = []
currentTime = 0

def main(argv):
    STDP = False
    duration = 1
    try:
      opts, args = getopt.getopt(argv,"sd:",["stdp"])
    except getopt.GetoptError:
      print ('main.py [-s, -d]')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-s", "--stdp"):
          STDP = True
      elif opt == '-d':
          duration = int(arg)

    global timestamps
    timestamps = np.arange(0, duration, timestep)
    #QuestionOne()
    #QuestionTwo(0)
    #QuestionTwo(-80)
    #PartB_QuestionOne()
    PartB_QuestionTwo(STDP)

if __name__ == "__main__":
   main(sys.argv[1:])
