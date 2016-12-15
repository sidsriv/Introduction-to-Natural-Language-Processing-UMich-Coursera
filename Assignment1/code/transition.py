# -*- coding: utf-8 -*-


class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """Add the arc (b, L, s) to A (arcs), and pop Σ (a stack).

        That is, draw an arc between the next node on the buffer and the next node on the stack, with the label L.

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        s = conf.stack[-1]
        # check that s is not the ROOT
        if s == 0:
            return -1
        # check that s is not already a dependant of some other word
        if any(s == wj for wi, l, wj in conf.arcs):
            return -1
        # take an item from the buffer, but do not pop it
        b = conf.buffer[0]
        conf.stack.pop()
        conf.arcs.append((b, relation, s))

    @staticmethod
    def right_arc(conf, relation):
        """Add the arc (s, L, b) to A (arcs), and push b onto Σ.

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        s = conf.stack[-1]
        # pop the buffer
        b = conf.buffer.pop(0)
        conf.stack.append(b)
        conf.arcs.append((s, relation, b))

    @staticmethod
    def reduce(conf):
        """Pop Σ (a stack).

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
            return -1
        s = conf.stack[-1]

        if any(s == wi for wk, l, wi in conf.arcs):
            conf.stack.pop()
        else:
            return -1

    @staticmethod
    def shift(conf):
        """Remove b from B (buffer) and add it to Σ (a stack).

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer:
            return -1

        conf.stack.append(conf.buffer.pop(0))
