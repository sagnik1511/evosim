# Implementation History
=========================================


This doc contains how I implemented the project from scracth with all the references I had to take on the way.


1. Creation of the Basic Game - 

- The basic game architecture was throught in brain and also some references and ideas came from [I programmed some creatures. They evolved. ft. davidrandallmiller](https://youtu.be/N3tRFayqVtk?si=M_Zhn-O24De7IVHp). An absolute gem.


2. Creating of basic functionalities between the elements and agents interacting in the environement. Some implementation ideation happened through [Kaggle-Lux-S1](https://github.com/Lux-AI-Challenge/Lux-Design-S1)


3. Working on the Policies - 
    a. I have implemented BasePolicy followed by PPO Policy. The policy is getting tested with diff configuration of the game env objects.

    To understand the policy I have take help from the original paper - [Proximal Policy Optimization Algorithm](https://arxiv.org/pdf/1707.06347) and ChatGPT for minor questionnaires.

     [code from nikhilbarhate](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py) also helped in the go.