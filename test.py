class candidate:
	def __init__(self,c,violations,observedProb,surfaceForm=None):
		self.c = c # This is the hidden structure, or the actual candidate that's got a violation profile to learn from
		self.violations = violations
		if surfaceForm is None:
			self.surfaceForm = c
		else:
			self.surfaceForm = surfaceForm # This is the output
		self.observedProb = observedProb
		self.harmony = 0
		self.predictedProb = 0
		self.checkViolationsSign()
	def checkViolationsSign(self): #convert all violations to negative numbers
		for i in range(0,len(self.violations)):
			# Convert to float - they'll probably be text when read from file
			# First, change blanks into 0's
			if self.violations[i]=="":
				self.violations[i]=0
			try:
				self.violations[i] = float(self.violations[i])
			except ValueError:
				print "It looks like some of your violations can't be interpreted as numbers.  Check whether there's some text in your violation profiles"
			if self.violations[i]>0:
				self.violations[i] = - self.violations[i]


class UR:
	def __init__(self,ur,prob=1,lexC=None):
		self.ur = ur
		self.prob = prob # Lexical frequency of this form
		# Check if the probability can be sensibly converted to a float
		try:
			self.prob = float(prob)
		except ValueError:
			print "It looks like your token probabilities can't all be converted to floats.  Check if there's some text in that column"
		self.candidates = []
		if lexC is None:
			self.lexC = [] # This can later be filled with tuples
		# e.g. (output, weight)
		# it's important that output is a surface form, not a hidden structure
		self.probDenom = 1 # denominator for calculating MaxEnt prob of each candidate
						   # this is the sum of each exponentiated harmony score
		self.predProbsList = []
		self.lastSeen = 0 # Keep track of what timestep the learner last saw this word
	
	def addCandidate(self,cand):
		self.candidates.append(cand)

	def decayLexC(self,t,decayRate,decayType='linear'):
		if len(self.lexC>1):
			if decayType=='linear':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=(t-self.lastSeen)*decayRate

			if decayType=='logarithmic':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=pow(t-self.lastSeen,decayRate)

			if self.lexC[i][1]<0: #If it's gotten as low as zero remove the constraint
				self.lexC.remove(self.lexC[i])

	def checkViolationLength(self): #check if all the violation vectors are the same length
		pass
	def calculateHarmony(self, w, t=None, decayRate=None, decayType=None): # Takes a vector of weights, equal in length to the violation vectors of the candidates
		if (decayRate is not None) and (t is None):
			sys.exit("You have to pass a time parameter to the calculateHarmony if you're using a decay rate")
		if decayRate is not None:
			self.decayLexC()
		for cand in self.candidates:
			# dot product
			cand.harmony = sum(viol*weight for viol,weight in zip(cand.violations,w))
			# Assuming the candidate's violations and weights are the correct sign, this'll be a negative number
			for i in self.lexC: # now add in stuff for the lexical constraint
				cand.harmony += i[1]*(0 if cand.surfaceForm==i[0] else 1)
			self.probDenom += pow(math.e,cand.harmony)
	def predictProbs(self,w):
		self.calculateHarmony(w)
		for cand in self.candidates:
			cand.predictedProb = pow(math.e,cand.harmony)/self.probDenom
	def getWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the highest harmony
		if theory =='MaxEnt': # sample from distribution
			rv_discrete() # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)

class Tableaux:
	def __init__(self,theory='MaxEnt'):
		self.urList = []
		self.urIDlist = []
		self.theory = theory
	def addUR(self,ur):
		self.urList.append(ur)
		self.urIDlist.append(ur.ur)






def readOTSoft(file):
	"""Reads in an OTSoft file type"""
	print 'reading tableaux...'
	tableaux = Tableaux()
	with open(file,"r") as f:
		lines=f.read().split('\r')
		lineno=0
		for line in lines:
			line=line.split('\t')
			print line
			if lineno==0:
				firstLine = line
				print firstLine
				if firstLine[1]=="":
					# OTSoft file? They have empty cells at the top
					pass
				if firstLine[0]=="input":
					# hgr file? They have the first three columns labeled
					inputType='hgr'
					print inputType
					# Constraints are in the first line too, so grab those
					# Headers of these files:
					# input output (hidden) probability (tab.prob) Constraint1 Constraint2 ...
					offset = (3 if firstLine[2]=='hidden' else 2)
					hidden = (True if firstLine[2]=='hidden' else False)
					tokenFrequency = (True if firstLine[offset+1]=='tab.prob' else False)
					offset = offset + (2 if firstLine[offset+1]=='tab.prob' else 1)
					constraints=firstLine[offset:]
					print offset
					print tokenFrequency
					print constraints
				else:
					print "I can't tell what type of file this is :("

			elif inputType=='hgr':
				ur = line[0]
				surfaceForm = (line[1] if hidden else None)
				c = (line[2] if hidden else line[1])
				observedProb = (line[3] if hidden else line[2])
				tokenProb = (line[offset-1] if tokenFrequency else 1)
				violations = line[offset:]
				print ur
				if ur not in tableaux.urIDlist:
					print 'add'
					tableaux.addUR(UR(ur,tokenProb))
				for i in tableaux.urList:
					if i.ur==ur:
						i.addCandidate(candidate(c,violations,observedProb,surfaceForm))
						break
			lineno+=1
			print lineno
	return tableaux
