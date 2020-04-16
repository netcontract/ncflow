from ..abstract_formulation import AbstractFormulation

class NCFlowAbstract(AbstractFormulation):

    @property
    def runtime(self):
        return self.runtime_est(14)

    def runtime_est(self, num_threads, breakdown=False):

        from heapq import heappush, heappop

        def heapsched_rt(lrts, k):
            h = []
            for rt in lrts[:k]:
                heappush(h, rt)

            curr_rt = 0
            for rt in lrts[k:]:
                curr_rt = heappop(h)
                heappush(h, rt + curr_rt)

            while len(h) > 0:
                curr_rt = heappop(h)

            return curr_rt

        def parallelized_rt(lrts, k):
            if len(lrts) == 0:
                return 0.0
            inorder_rt = heapsched_rt(lrts, k)
            cp_bound = max(lrts)
            area_bound = sum(lrts) / k
            lrts.sort(reverse=True)
            two_approx = heapsched_rt(lrts, k)

            if self.VERBOSE:
                self._print("-- in incoming order, schedule= ", inorder_rt)
                self._print("-- bounds cp= ", cp_bound, "; area= ", area_bound)
                self._print("-- sorted rts: ", lrts)
                self._print("-- in sorted order, schedule ", two_approx)

            return two_approx

        rts = self._runtime_dict
        r2_time = parallelized_rt(list(rts['r2'].values()), num_threads)
        reconciliation_time = parallelized_rt(
            list(rts['reconciliation'].values()), num_threads)

        if 'kirchoffs' in rts:
            kirchoffs_time = parallelized_rt(
                list(rts['kirchoffs'].values()), num_threads)
        else:
            kirchoffs_time = 0

        print('Runtime breakdown: R1 {} R2// {} Recon// {} R3 {} Kirchoffs// {} #threads {}'.format(
            rts['r1'], r2_time, reconciliation_time, rts['r3'], kirchoffs_time, num_threads))
        if breakdown:
            return rts['r1'], r2_time, reconciliation_time, rts['r3'], kirchoffs_time

        total_time = rts['r1'] + r2_time + reconciliation_time + rts['r3'] + kirchoffs_time

        return total_time
