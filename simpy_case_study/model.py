import simpy
import numpy as np
from collections import namedtuple
from simpy_case_study.helpers import SamplingHelpers, StaticParameter, Targets


class Counter:
    def __init__(self, func):
        self.count = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(job_id=self.count, *args, **kwargs)


def job_source(env, product_mixture, **kwargs):
    while True:
        # sample from job mixture
        job_type = np.random.choice(len(product_mixture), p=product_mixture)
        job = process_job(env, job_type, **kwargs)
        env.process(job)
        timeout = SamplingHelpers.expon_dist(loc=StaticParameter.JOB_INTERVALL)
        yield env.timeout(timeout)


@Counter
def process_job(env, job_type, job_id, machines, buffer, testing_station, targets):
    start_time = env.now
    with buffer.request() as buffer_req:
        yield buffer_req
        # machine processing
        machine = yield machines.get()

        # job type specific factors for processing time mean & variance
        if job_type == 0:
            processing_time_factor = 1
            processing_time_variance = 1
        elif job_type == 1:
            processing_time_factor = 3
            processing_time_variance = 1
        elif job_type == 2:
            processing_time_factor = 7
            processing_time_variance = 5
        processing_time = SamplingHelpers.gaussian_dist(
            loc=machine.proc_time_mean * processing_time_factor,
            scale=processing_time_variance
        )
        yield env.timeout(processing_time)
        # BEVOR MASCHINE WIEDER FREI IST, CHECKEN OB NACHFOLGENDER PUFFER BEREIT IST!

        yield machines.put(machine)

        # quality control:
        with testing_station.request() as req:
            yield req
            testing_time = SamplingHelpers.gaussian_dist(loc=StaticParameter.TESTING_TIME_MEAN)
            yield env.timeout(testing_time)

            # does job fails quality control?
            if SamplingHelpers.uniform_dist() <= StaticParameter.ERROR_RATE:
                pass
            else:
                end_time = env.now
                targets.CYCLE_TIME.append(end_time - start_time)
                drain(env)


@Counter
def drain(env, job_id):
    pass


# !!!! EIN DECORATOR ZUM LOGGEN EINES RUNS (DER ALLE PRINTS ABFÄNGT BZW DER EINE LOG LISTE ABFÄNGT)
def simulation_run(factor_config: object, noise_config: object):
    """
    returns target
    """
    targets = Targets()
    product_mixture = noise_config.product_mix
    product_mixture = list(map(lambda x: x / sum(product_mixture), product_mixture))
    for repl in range(StaticParameter.REPLICATIONS):
        Machine = namedtuple('Machine', 'proc_time_mean')
        machines = [Machine(proc_time_mean) for proc_time_mean in StaticParameter.MACHINE_PROCESSING_MEAN]
        machines = machines[:factor_config.num_machines]
        process_job.count = 0
        drain.count = 0
        env = simpy.Environment()
        buffer = simpy.Resource(env, capacity=factor_config.buffer_size)
        machine_shop = simpy.FilterStore(env, capacity=factor_config.num_machines)
        machine_shop.items = machines
        testing_station = simpy.Resource(env, capacity=factor_config.num_testing_station)
        env.process(job_source(env, machines=machine_shop, buffer=buffer, testing_station=testing_station,
                               product_mixture=product_mixture, targets=targets))
        env.run(until=StaticParameter.SIM_TIME)
        targets.THROUGHPUT.append(drain.count)
        targets.CYCLE_TIME.append(sum(targets.CYCLE_TIME) / len(targets.CYCLE_TIME))

    # print(f"mean throughput: {sum(targets.THROUGHPUT) / len(targets.THROUGHPUT)}")
    # print(f"mean cycle time: {sum(targets.CYCLE_TIME) / len(targets.CYCLE_TIME)}")
    return sum(targets.THROUGHPUT) / len(targets.THROUGHPUT)
