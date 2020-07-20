# Discrimination Mitigation

## Introduction
Increasingly, organizations utilize big data 
to guide human resource decision making, 
often through machine-learning predictive analytics. 
Decisions based on predictive models risk 
running afoul of ethical concerns, 
including those related to unintended discrimination
against protected classes and other groups.
A growing literature in "AI Fairness" 
seeks to address these concerns, 
defining many alternatively mathematical formulations of unfairness
as well as remedial algorithms, 
many of which are novel. 
The foundational ethical concerns, however, 
are not. 
 
In this white paper, 
we argue in favor of a simple algorithm  
for mitigating potential discrimination
against arbitrary group members
on the basis of model predictions. 
Motivated by the Frisch-Waugh-Lovell theorem,
our algorithm applies to predictive models generally 
and can be applied after the models are already trained -- 
so long as the estimated models included
relevant group variables as predictors. 
We have implemented this algorithm in a python package: 
`DiscrminationMitigator`.
The current version is compatible with Keras and GBM 
models, including those estimated through `FIFE`. 

Our argument is motivated
by the economic literature on *statistical discrimination*,
which is defined as basing a decision on group membership
because group memberships proxies 
for other unobserved but relevant traits. 
In essence, in a predictive model
that excludes group membership, 
predictors can be decomposed into a component
that predicts the target holding group membership constant
and a component that predicts the target
because it predicts group membership.  
Decisions based on the former might 
cause a disparate impact, 
but only decisions based on the latter are problematic. 

Our argument pertains directly to a central 
dilemma of the many Supreme Court rulings
on the Civil Rights Act of 1964. 
Namely, decision rules that favor one group 
over another in aggregate (*disparate impact*)
automatically stand in violation of the civil rights act;
but it is unrealistic to expect that reasonable hiring criteria 
will not have a disparate impact, so these can excused 
if a sufficiently convincing argument of *business necessity* can be made.
Although *business necessity* is legal jargon, 
it is worth pointing out that critera underlying human resource decisions 
are rarely business necessities in their own right;  
in their own right;
rather, they are predictors of actual business necessities, 
such as workplace performance. 
Part -- and only part -- of a predictor 
operates through statistical discrimination 
against omitted groups. 
It is this part 
that should be excluded 
from human resource decision making processes.   
  
  
### Literature Review

#### supreme court rulings
Even when superficially blind to group membership, 
human resource choices based on personal attributes 
can, in aggregate, favor one group over another. 
In rulings from Griggs v. Duke Power to Ricci v. DeStefano, 
the Supreme Court has demarcated where such choices 
are permissible,  
and where they violate the Civil Rights Act of 1964 
(and amendments in 1991).

These rulings dance around a central dilemma.
At root, covert discriminatory purposes can be achieved 
by making human resource decisions 
on the basis of superficially neutral criteria
that differentially affect groups. 
Because intent can be obscured so easily, 
the Court's only practical countermeasure 
is treat any criterion as discriminatory if 
its consequence is a *disparate impact*, 
or disproportionate effect to certain group members.
But to apply this rule categorically  
would be hopelessly impractical.  

[A second dilemma often follows from the fist: 
avoiding one flavor of covert discrimination
often required a different flavor of overt discrimination.]  

Although this white paper does not speak directly to jurisprudence,
a critical discussion of past rulings will clarify 
the statistical nature of these dilemmas -- 
as well as its statistical solution. 
Whereas past rulings tend to be impaled by one or the other, 
our white paper grasps the dilemma by the horns. 


 many close rulings.  Majority opinion takes one side of the dilemma, and 

Griggs V. Duke. majority opinion.
laid out disparate impact,
Inflexible application of this criterion is absurd.  For example, 
But 
business necessity.  
While it may be clear that 
a medical degree a pre-requiste to practicing medicine, 
other certifications and personal attributes occupy a middle ground.  
strenght 
Forbidding disparate impact was never an end in itself.
Rather, it was a means to prevent covert discrimination.   
 

Ricci v. Steffano - FD violated Title VII of 1964. 
* FD had gone to lengths to ensure fairness and relevance. -- like a business necessity 
* outcome had disparate impact
* FD threw out test
* Kennedy - disparate treatment "express, raced-based decisionmaking"
* city's objective was to avoid disparate impact liability. 







 
 * Labor market decision based on personal attributes will, in aggregate, have different
 effects on different groups.  
 * However, entirely diregarding personal attributes is not viable.
 .. keep above short.  elaborate below when discussing rulings.  
 
 To thread this needle, supreme court (and latter congress in amendment) 
 In essence, some criteria are elevated to   

  
  


* Methodology is more broadly applicable than to protected classes, 
but discussion about protected classes is informative.   

Original supreme court ruling - ostensibly race neutral, but 

Shortcoming -- how many surgeons must be unemployed?  

band-aid ruling: "business necessity"

But are the credentials business necessities at all? 
Or are they simply predictors of future job performance, 
which itself is the business necessity?  

when something predicts 
credentials and other predictors of job performance predict
through two channels -- two types of 


 
and also predicts an outcome


#### statistical discrimination 

* review econ distinction.  
* machine learning problem. 
* statistically discriminates^2 -> 
** black to statistically discrminate against possess
** cornrows to statistically discriminate against blacks 

Proxy discrmination (sometimes used differently elsewhere)

Any variable 


#### 'fairness' criteria
While our paper fits clearly in with economic literature of statistical discrimination
and speaks directly to jurisprudence,  

We briefly assess this literature in appendix _. 
___ review recent papers.
  
Practical problems: improving the 'fairness' of model s 


deeper philosophical problem of fairness?
(1) distinguishing between the forecast and the decision rule.  
(2) different criteria are mutually irreconcilable
(3) underpinnings -- social planner cares about aggregate welfare.
might care about inequality, but this is because inequality reduces 
aggregate welfare through concavity of the utility function.  

Discrimination on the basis of group membership itself does not prose an ethical
problem, or else all decisionmaking does -- as any predictors can be converted into 
group indicators (e.g. low iq people, high iq people).  Discrmination on the basis of ethnicity 
is also not wrong in itself, or else it would be wrong for a doctor to tailor screenings
to account for ethnic risks (e.g. Niemann-Pick, cystic fibrosis, sickle-cell). 
Rather, racism itself is wrong and causes enormous damage to society.  
Expectations of discrmination can reduce the incentives to invest in human capital, 
which in turn reinforce the group differences that drive statistical discrimination.
(keven hasset? who was kristin's boss?)   
Discrimination on the basis of race -- whether taste-based or statistical -- 
aggravates prior damage. 
 


For what purpose should a person care about the fairness criteria?
Legislate?  Legal?  benevolence?    



it fits within a broader discussion 
of machine learning 'fairness criteria'.  


### move!
 

The machine-learning 'fairness' literature is wide; 
a large number of fairness criteria have been proposed, 
but there has been little follow-up on particular methods.



The fairness criteria have many shortcomings: 
1. Generically, different fairness criteria cannot be satisfied simultaneously. 
2. Fairness criteria often conflate prediction with decision rules, which 
tend to preclude their application to frameworks 
in which the decision rules are contingent or unknown.  
3. Algorithms intended to satisfy fairness criteria often require special
transformation of the data used to train the model or require customization 
of the estimation procedure, such as through the loss function.  
Consquently, such fairness criteria cannot be implemented in a model that
has already been trained.  
4. Many algorithms intended to satisfy fairness criteria are limited to 
particular  


Below we discuss several fairness criteria with the aim of illustrating the
above problems.  This discussion is not comprehensive; instead we would point 
readers to _____.  

[discussion of several fairness criteria ]

The methodology we propose address statistical discrimination, which we 
believe to be the most pernicious and also the most rectifiable risk of 
machine learning models.  In essence, 

#### statistical discrmination
Classically, economists consider two types of discrmination: taste-based disrimination
and statistical discrmination. [cite]


definition taste based

definition statistical 

* why this distinction is relevant

- a user without animus might nevertheless engage 
in statistical discrimination because
doing so is beneficial to his objective. 

- a model that predicts an outcome relevant to a decision maker
however, 

* taste-based discrimination is not totally irrelevant, however.  That is, taste-based 
discrimination can generate bias in the target or covariates used to 
train the model.  For example, promotion decisions might be biased against, 

classically, statistical discrimination occurs when a decisionmaker 
overtly uses the predictive content of protected class.  

However, excluding protected class from a prediction model is not sufficient
to prevent it from discriminating.  

### Proxy Discrimination and Frisch-Waugh-Lovell
[discussion written in lyx]

## discussion of algorithm with illustrations 


### base case

### consequence - downweighting of proxy predictors


### shortomings 
   
 good faith in the decision rule will still be important
 this methodology does not insulate predictions from the consequences of 
 structural discrimination.    