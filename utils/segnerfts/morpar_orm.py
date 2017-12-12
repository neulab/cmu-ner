#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import re, collections, functools, json
from copy import deepcopy
from argparse import Namespace

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
    
################################
#
# Utility classes
#
################################
        
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator
    
class HashableDict(dict):
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
        
    def __lshift__(self, other):
        result = HashableDict()
        for key in list(self) + list(other):
            if key in result:
                continue
            if key not in self:
                result[key] = other[key]
            elif key not in other:
                result[key] = self[key]
            else:
                result[key] = self[key] >> other[key]
                    
        return result
        
    def __rshift__(self, other):
        result = HashableDict()
        for key in list(self) + list(other):
            if key in result:
                continue
            if key not in self:
                result[key] = other[key]
            elif key not in other:
                result[key] = self[key]
            else:
                result[key] = self[key] << other[key]
        return result    
    
             
        
##################################
#
# CHANNELS
#
##################################
       
class Channel(object):   
        
    def __init__(self, name):
        self.name = name
        
    def __eq__(self, other):
        if other == None:
            return False
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
        
    def __call__(self, str):
        return Default(str, self)
        
    def __str__(self):
        return self.name
        
    def __repr__(self):
        return str(self)
        
    def __div__(self, other):
        if isinstance(other, LiteralParser) or isinstance(other, PatternParser):
            return other.rechannel(self) + other
        return ChannelSequence(self, other)
        
    def __contains__(self, item):
        return False
        
    def __le__(self, other):
        return self == other or self in other
        
    def __iter__(self):
        yield self
        
    def __len__(self):
        return 1
        
    def typIsStr(self):
        return isinstance(self.typ(), unicode)
        
    def join(self, others):
        return self.typ( self.typ().delimiter().join(others) )
    
class ChannelSequence(Channel):

    def __init__(self, l_child, r_child):
        self.l_child = l_child
        self.r_child = r_child
        
    def __hash__(self):
        return hash(str(self))
        
    def __call__(self, str):
        return self.l_child(str) + self.r_child(str)
    
    def __le__(self, other):
        return self.l_child <= other and \
               self.r_child <= other  
        
    def __eq__(self, other):
        if other == None:
            return False
        return self <= other and other <= self
        
    def __contains__(self, item):
        return item == self.l_child or \
               item in self.l_child or \
               item == self.r_child or \
               item in self.r_child
               
    def __str__(self):
        return "%s+%s" % (self.l_child, self.r_child)
   
    def __iter__(self):
        for x in self.l_child:
            yield x
        for y in self.r_child:
            yield y
            
    def __len__(self):
        return len(self.l_child) + len(self.r_child)
    
    def typIsStr(self):
        return self.l_child.typIsStr() and self.r_child.typIsStr()
        
class AbstractPatternTyp(object):

    def __init__(self, pattern, typ):
        self.pattern = pattern
        self.typ = typ
        
    def is_pattern(self):
        return True
        
    def __lshift__(self, other):
        return self.replacePattern(other)
        
    def __rshift__(self, other):
        return AbstractPatternTyp("%s%s%s" % (self.pattern, self.typ().delimiter(), other), self.typ)       
        
    def replacePattern(self, input):
        # self is the pattern, input is what goes into it
        
        if not hasattr(self, 'outputPattern'):
        
            backwards_pattern = createVariablePattern(self.pattern, self.typ().delimiter())
            self.backwards_regex = re.compile(backwards_pattern)
            self.outputPattern = createNumberedPattern(self.pattern)
            
        text_out = self.backwards_regex.sub(self.outputPattern, input, count=1)
        return self.typ(text_out)
        
    def __str__(self):
        return "/" + self.pattern + "/"
        
    def __repr__(self):
        return str(self)
        
class AbstractStr(unicode):
    ''' This is a subclass of str that string channels' typs descend from; it defines
        some default behavior for concatenation, testing for prefixation/suffixation, etc. '''

    def __lshift__(self, other):
        if other.is_pattern():
            return AbstractPatternTyp("%s%s%s" % (self, self.delimiter(), other), other.typ)
        return type(self)("%s%s%s" % (self, self.delimiter(), other))
        
    def __rshift__(self, other):
        if other.is_pattern():
            return other.replacePattern(self)
        return type(self)("%s%s%s" % (self, self.delimiter(), other))
        
    def endsWith(self, other):
        comparison_form = self.rstrip(self.delimiter())
	r1 = re.compile(other + "$")	
	return r1.search(comparison_form)
        #return comparison_form.endswith(other)
    
    def startsWith(self, other):
        comparison_form = self.lstrip(self.delimiter())
        r1 = re.compile("^" + other)
	return r1.search(comparison_form)
	#return comparison_form.startswith(other)
    
    def hasPrefix(self, other):
        if self == other:
            return True
        pref = self[:len(other)+len(self.delimiter())]
        return pref == other + self.delimiter()
        
    def hasSuffix(self, other):
        if self == other:
            return True
        suf = self[-len(other)-len(self.delimiter()):]
        return suf == self.delimiter() + other

    def stripPrefix(self, other):
        assert(self.hasPrefix(other))
        if self == other:
            return type(self)('')
        return Concatenated.typ(self[len(other)+len(self.delimiter()):])
    
    def stripSuffix(self, other):
        assert(self.hasSuffix(other))
        if self == other:
            return type(self)('')
        return type(self)(self[:-len(other)-len(self.delimiter())])

    def is_pattern(self):
        #return self[:1] == '/' and self[-1:] == '/'
        return False
        
        
#######################################
#
# PARSERS
# 
########################################

class Parser(object):
    
    def __init__(self, channel=None):
        self.channel = channel
        
    def get_channel(self):
        return self.channel
        
    def _is_trivial(self, input_channel):
        #print(self.get_channel(), input_channel)
        return not(set(self.get_channel()) & set(input_channel))
    
    def __lshift__(self, other):
        assert(isinstance(other, Parser))
        return RightwardSequence(self, other)
        
    def __rshift__(self, other):
        assert(isinstance(other, Parser))
        return LeftwardSequence(self, other)
        
    def __add__(self, other):
        assert(isinstance(other, Parser))
        return Sequence(self, other)
        
    def __sub__(self, other):
        assert(isinstance(other, Channel))
        return Trim(self, other)
        
    def __or__(self, other):
        return Choice(self, other)
        
    def __invert__(self):
        return Negation(self)
        
    def constructOutput(self, parse_result, remnant):
        return set([(parse_result, remnant)])
        
    def constructEmptyOutput(self, remnant):
        return set([(HashableDict(), remnant)])
        
    def _trivial_parse(self, input, input_channel, leftward=False):
        return self.constructEmptyOutput(input)   # a trivial success 
        
    def _nontrivial_parse(self, input, input_channel, leftward=False):
        return self.constructEmptyOutput(input)  # a trivial success
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
    
        if input_channel == None:  # assign it here rather than in the function definition
            input_channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
        if self._is_trivial(input_channel):
            return self._trivial_parse(input, input_channel, leftward)
            
        return self._nontrivial_parse(input, input_channel, leftward)
    
    
    def parse(self, s, input_channel=None):
        ''' Parse the str into a list of outputs.  A wrapper around __call__ that discards
            incomplete parse outputs (that is, ones that have a remainder) '''
        
        if input_channel == None:  # assign it here rather than in the function definition
            input_channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
        if len(input_channel) > 1:
            print("ERROR: parse method requires a simple data type; %s is complex." % input_channel)
            return []
        if not input_channel.typIsStr():
            print("ERROR: Cannot parse a non-string data type: %s" % input_channel)
            return []
            
        input = HashableDict()
        for input_channel in input_channel:
            input[input_channel.name] = input_channel.typ(s)
        parses = self(input, input_channel)
        return [output for output, remnant in parses if not remnant[input_channel.name]]
 
            
class LiteralParser(Parser):

    def __init__(self, pattern, channel=None):
        super(LiteralParser, self).__init__(channel)
        assert(len(channel)==1)
        self.text = pattern
        self.pattern = channel.typ(pattern)
        self.output = HashableDict({channel.name:channel.typ(pattern)})
        
    def rechannel(self, channel):
        return Lit(self.text, channel)
        
    def _trivial_parse(self, input, input_channel=None, leftward=False):
        return self.constructOutput(self.output, input)
        
    def _nontrivial_parse(self, input, input_channel=None, leftward=False):
        text = input[self.channel.name]
            
        hasAffix = text.hasPrefix if leftward else text.hasSuffix
        stripAffix = text.stripPrefix if leftward else text.stripSuffix
        
        if hasAffix(self.pattern):
            remnant = deepcopy(input)
            remnant[self.channel.name] = stripAffix(self.pattern)
            return self.constructEmptyOutput(remnant)
        else:
            return set()
    
    
class Guess(Parser):

    def __init__(self, channel=None, output_channels=None):
        super(Guess, self).__init__(channel)
        self.output_channels = output_channels if output_channels else channel
    
    def _nontrivial_parse(self, input, input_channel=None, leftward=False):
        
        assert(len(input_channel)==1)
        
        results = set()
        text = input[input_channel.name]        

        hasAffix = text.hasPrefix if leftward else text.hasSuffix
        stripAffix = text.stripPrefix if leftward else text.stripSuffix
                
        for i in range(len(text)):
            remnant = deepcopy(input)
            substr = text[:i+1] if leftward else text[i:]
            stem = input_channel.typ(substr)
            if hasAffix(stem):
                remnant[input_channel.name] = stripAffix(stem)
                output = HashableDict()
                for output_channel in self.output_channels:
                    if output_channel != input_channel:
                        output[output_channel.name] = output_channel.typ(stem)
                results.add((output, remnant))
        return results

class BinaryCombinator(Parser):

    def __init__(self, l_child, r_child):
        self.l_child = l_child
        self.r_child = r_child
    
    def get_channel(self):
        return self.l_child.get_channel() & self.r_child.get_channel()

        
        
class Sequence(BinaryCombinator):
    ''' A parser that executes its children in sequence, and applying the second to the remnant of the first.  The direction (left child first or right child first) depends on the value passed into the parameter leftward. '''

    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
            
        child1 = self.l_child if leftward else self.r_child
        child2 = self.r_child if leftward else self.l_child
        
        results = set()
        for outputs1, remnant1 in child1(input, input_channel, leftward):
            for outputs2, remnant2 in child2(remnant1, input_channel, leftward):
                outputs = outputs1 >> outputs2 if leftward else outputs2 << outputs1
                results.add((outputs, remnant2))
        return results
            
class RightwardSequence(Sequence):
    ''' This is a sequence combinator that always executes right-to-left, regardless
        of the directionality of higher combinators. (That is to say, it ignores the
        passed-in value of the leftward parameter and always acts as if it's False.) '''
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        return super(RightwardSequence, self).__call__(input, input_channel, False)  
  
class LeftwardSequence(Sequence):
    ''' This is a sequence combinator that always executes left-to-right, regardless
        of the directionality of higher combinators. (That is to say, it ignores the
        passed-in value of the leftward parameter and always acts as if it's True.)'''
    
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        return super(LeftwardSequence, self).__call__(input, input_channel, True)
        
class Choice(BinaryCombinator):

    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        return self.l_child(input, input_channel, leftward) | self.r_child(input, input_channel, leftward)

        
class AssertParser(Parser):
    ''' An Assert parser asserts a predicate of the input at that stage 
        of the parse and failing (that is, returning set([])) if the predicate fails '''

    def __init__(self, pred, channel=None):
    
        if channel == None:  # assign it here rather than in the function definition
            channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
        self.channel = channel
        self.pred = pred
        
    def _is_trivial(self, input_channel):
        return True
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
            
        if self.channel.name not in input or not self.channel <= input_channel:
            return self.constructEmptyOutput(input)
       
        text = input[self.channel.name]
        if not self.pred(text):
            return set()
            
        return self.constructEmptyOutput(input)
        
class Delay(Parser):
    ''' A Delay allows you to make recursive grammars.  If you need to refer to a parser that you have not yet defined, you will get
        a NameError in Python.  To avoid this, you can refer to a label as Delay(lambda:X) rather than X. '''
        
    def __init__(self, parser):
        self.parser = parser
        
    def get_channel(self):
        return self.parser().get_channel()
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        return self.parser()(input, input_channel)    
        
        
class Negation(Parser):

    def __init__(self, child):
        self.child = child
    
    def _is_trivial(self, input_channel):
        return self.child._is_trivial(input_channel)
    
    def get_channel(self):
        return self.child.get_channel()
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        child_results = self.child(input, input_channel, leftward)
        if child_results:
            return set()
        return self.constructEmptyOutput(input)
        
        
class NullParser(Parser):

    def __init__(self):
        super(NullParser, self).__init__([])
       
def generateGroup(x=0):
    while True:
        x += 1
        yield "\%s" % x

def createNumberedPattern(pattern, start_number=1):

    char_index = 0
    paren_count = 0
    result = ''
    while char_index < len(pattern):
        if pattern[char_index] == '\\':
            if paren_count == 0:
                result += pattern[char_index:char_index+1]
            char_index += 2
            continue
        if pattern[char_index] == '(':
            paren_count += 1
            char_index += 1
            continue
        if pattern[char_index] == ')':
            paren_count -= 1
            char_index += 1
            if paren_count == 0:
                result += '\\%s' % start_number
                start_number += 1
            continue
        if paren_count == 0:
            result += pattern[char_index]
        char_index += 1
        
    return result
    
def getGroupsFromPattern(pattern):

    char_index = 0
    paren_count = 0
    results = []
    currentResult = ''
    while char_index < len(pattern):
        if pattern[char_index] == '\\':
            if paren_count > 0:
                currentResult += pattern[char_index:char_index+1]
            char_index += 2
            continue
        if pattern[char_index] == '(':
            paren_count += 1
            char_index += 1
            currentResult += '('
            continue
        if pattern[char_index] == ')':
            paren_count -= 1
            char_index += 1
            currentResult += ')'
            if paren_count == 0:
                results.append(currentResult)
                currentResult = ''
            continue
        if paren_count > 0:
            currentResult += pattern[char_index]
        char_index += 1
    return results
    
def createVariablePattern(pattern, delimiter):

    results = getGroupsFromPattern(pattern)
    return delimiter.join(results) 

class RedupParser(Parser):

    def __init__(self, pattern, channel=None, base_pattern=None):
    
        if channel == None:
            channel = DEFAULTS.Text
        
        self.channel = channel
        self.pattern = pattern
        if not base_pattern:
            base_pattern = pattern
        
        self.leftward_regexes = {}
        self.rightward_regexes = {}
        
        self.output = HashableDict()
        for chn in self.channel:
            
            leftward_pattern = "(" + pattern + ")" + chn.typ().delimiter() + "(" + createNumberedPattern(base_pattern, 2) + ".*)"
            self.leftward_regexes[chn] = re.compile(leftward_pattern)
            
            rightward_pattern = "(.*" + base_pattern + ")" + chn.typ().delimiter() + "(" + createNumberedPattern(pattern, 2) + ")"
            self.rightward_regexes[chn] = re.compile(rightward_pattern)
            
            output_pattern = pattern + chn.typ().delimiter() + base_pattern
        
            self.output[chn.name] = chn.pattern_typ(output_pattern)
        
    def rechannel(self, channel):
        return Pattern(self.pattern, channel)
    
    def _trivial_parse(self, input, input_channel=None, leftward=False):
        return self.constructOutput(self.output, input)
    
    def _nontrivial_parse(self, input, input_channel=None, leftward=False):
        text_in = input[input_channel]
        if leftward:
            match = self.leftward_regexes[input_channel].match(text_in)
        else:
            match = self.rightward_regexes[input_channel].match(text_in)
        if not match:
            return set()
        remnant = deepcopy(input)
        if leftward:
            remnant[input_channel.name] = input_channel.typ(match.groups()[-1])
        else:
            remnant[input_channel.name] = input_channel.typ(match.group(1))
            
        result = HashableDict()
        for chn in self.channel:
            if chn == input_channel:
                continue
            if leftward:
                result[chn] = chn.typ(match.group(1))
            else:
                result[chn] = chn.typ(match.groups()[-1])
        
        return self.constructOutput(result, remnant)
    
class PatternParser(Parser):

    def __init__(self, pattern, channel=None):
    
        if channel == None:
            channel = DEFAULTS.Text
        assert(len(channel)==1)
        
        self.channel = channel
        self.pattern = pattern
	self.parse_regex = re.compile(pattern + r"$")
        #self.outputPattern = createNumberedPattern(pattern)
        self.output = HashableDict({
            self.channel.name:
                self.channel.pattern_typ(pattern)
        })
        
    def _trivial_parse(self, input, input_channel=None, leftward=False):
        return self.constructOutput(self.output, input)
        
    def _nontrivial_parse(self, input, input_channel=None, leftward=False):
        text_in = input[self.channel.name]
        match = self.parse_regex.match(text_in)
        if not match:
            return set()
        remnant = deepcopy(input)
        remnant[input_channel.name] = self.channel.join(match.groups())
        return self.constructEmptyOutput(remnant)
        
    def rechannel(self, channel):
        return Pattern(self.pattern, channel)
         

class Trim(Parser):

    def __init__(self, child, channel):
        self.child = child
        self.channel = channel
        
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
    
        if input_channel == None:  # assign it here rather than in the function definition
            input_channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
        results = set()
        for child_output, child_remnant in self.child(input, input_channel, leftward):
            output = deepcopy(child_output)
            for channel in self.channel:
                if channel.name in output:
                    del output[channel.name]
            results.add((output, child_remnant))
        return results  

class Truncate(Parser):

    def __init__(self, text, channel):
        self.text = text
        self.channel = channel
        
    def __call__(self, input, input_channel=None, leftward=False):
    
        if input_channel == None:  # assign it here rather than in the function definition
            input_channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
          
        if self._is_trivial(input_channel):
            return self.constructEmptyOutput(input)
            
        remnant = deepcopy(input)
        if leftward:
            remnant[input_channel.name] = input_channel.typ(self.text) >> remnant[input_channel.name]
        else:
            remnant[input_channel.name] = remnant[input_channel.name] << input_channel.typ(self.text)
        return self.constructEmptyOutput(remnant)
        
#####################################
#
# Convenience functions for channels
#
#####################################


def make_channel_from_delimiter(delim):

    class AnonymousChannel(Channel):
        class typ(AbstractStr):
            def delimiter(self):
                return delim
                
        def pattern_typ(self, x):
            return AbstractPatternTyp(x, self.typ)
                
    return AnonymousChannel
    
            
###############################
#
# Convenience functions
#
###############################

def Default(text, channels=None):
    if channels == None:
        channels = DEFAULTS.Text
    if getGroupsFromPattern(text):
        return Pattern(text, channels)
    return Lit(text, channels)
    
    
def Pattern(pattern, channels=None):

    if channels == None:  # assign it here rather than in the function definition
        channels = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
    return make_multichannel_parser(PatternParser, pattern, channels)

    
def Lit(text, channels=None):
    if channels == None:
        channels = DEFAULTS.Text
    return make_multichannel_parser(LiteralParser, text, channels)
    
    
    
def make_multichannel_parser(parser, pattern, channels=None):

    if channels == None:
        channels = DEFAULTS.Text

    result = None
    for channel in channels:
        p = parser(pattern, channel)
        result = p if not result else Sequence(result, p)
    return result
        
        
    
def Assert(pred, channel=None):

    if channel == None:  # assign it here rather than in the function definition
        channel = DEFAULTS.Text   # in case the library user redefines the concatenation type of Text
            
    return make_multichannel_parser(AssertParser, pred, channel)
        
def After(s, channel=None):
    pred = lambda x: x.endsWith(s)
    return Assert(pred, channel)
    
def Before(s, channel=None):
    pred = lambda x: x.startsWith(s)
    return Assert(pred, channel)

    
NULL = NullParser()


######################################
#
# BUILT-IN CHANNELS
#
######################################

Concatenated = make_channel_from_delimiter("")
Spaced = make_channel_from_delimiter(" ")
Hyphenated = make_channel_from_delimiter("-")
        
Tex = Concatenated("text")
Mor = Hyphenated("breakdown")
Lem = Hyphenated("lemma")
Glo = Hyphenated("gloss")
Cit = Hyphenated("citation")
All = Tex/Mor/Lem
    
DEFAULTS = Namespace(Text=Tex, AllChannels=All)
   
