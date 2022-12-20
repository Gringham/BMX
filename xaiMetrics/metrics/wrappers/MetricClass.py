import abc

from xaiMetrics.constants import REFERENCE_BASED, REFERENCE_FREE


class MetricClass(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for metrics
    '''

    @abc.abstractmethod
    def __call__(self, ref, hyp):
        '''
        This function calculates a metric given all of its parameters (ref could also be src)
        :return: score
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        '''
        This method returns an object holding information about the current configuration
        :return: ConfigurationVersionString
        '''
        raise NotImplementedError

    def evaluate_df(self, df):
        if self.mode == REFERENCE_BASED:
            return self.__call__(df['REF'].tolist(), df['HYP'].tolist())
        elif self.mode == REFERENCE_FREE:
            return self.__call__(df['SRC'].tolist(), df['HYP'].tolist())