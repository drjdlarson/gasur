# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 07:50:04 2020

@author: ryan4
"""


class IncorrectNumberOfTargets(Exception):
    def __init__(self, expected, given):
        msg = "Number of targets do not match.\n"
        msg += "\tExpected: {exp:d}.\n\tGiven: {giv:d}".format(exp=expected,
                                                               giv=given)
        Exception.__init__(self, msg)
