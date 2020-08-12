import simpy
import numpy as np
from collections import namedtuple
from case_study.helpers import SamplingHelpers

test = 7

class Counter:
    def __init__(self, func):
        self.count = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(job_id= self.count, *args, **kwargs)

def job_source(env, interval, **kwargs):
    while True:
        #sample from job mixture
        job_type = np.random.choice(len(Noise.PRODUCT_MIXTURE), p= Noise.PRODUCT_MIXTURE)
        job = process_job(env, job_type, **kwargs)
        env.process(job)
        timeout = SamplingHelpers.expon_dist(loc= Parameters.JOB_INTERVALL)
        yield env.timeout(timeout)
        
@Counter
def process_job(env, job_type, job_id, machines, buffer, testing_station):
    start_time = env.now
    log.append(f"{start_time}: job {job_id} of type {job_type} arrived")
    with buffer.request() as buffer_req:
        yield buffer_req
        #machine processing
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
        processing_time = get_gaussian_dist_number(
            loc= machine.proc_time_mean * processing_time_factor,
            scale= processing_time_variance
        )
        yield env.timeout(processing_time)
        #BEVOR MASCHINE WIEDER FREI IST, CHECKEN OB NACHFOLGENDER PUFFER BEREIT IST!

        yield machines.put(machine)
            
        #quality control:
        with testing_station.request() as req:
            yield req
            testing_time = get_gaussian_dist_number(loc= Parameters.TESTING_TIME_MEAN)
            yield env.timeout(testing_time)

            #does job fails quality control?
            if get_uniform_dist_number() <= Parameters.ERROR_RATE:
                pass
            else:
                end_time = env.now
                cycle_time.append(end_time - start_time)
                drain(env)

@Counter
def drain(env, job_id):
    pass


#!!!! EIN DECORATOR ZUM LOGGEN EINES RUNS (DER ALLE PRINTS ABFÄNGT BZW DER EINE LOG LISTE ABFÄNGT)
def simulation_run(factor_config : object, noise_config : object, replications : int = 1):
    '''
    returns target
    '''
    throughputs = []
    cycle_times = []
    for repl in range(replications):
        Machine = namedtuple('Machine', 'proc_time_mean')
        machines = [Machine(proc_time_mean) for proc_time_mean in Parameters.MACHINE_PROCESSING_MEAN]
        machines = machines[:Parameters.NUM_MACHINES]
        process_job.count = 0
        drain.count = 0
        env = simpy.Environment()
        buffer = simpy.Resource(env, capacity= Parameters.BUFFER_SIZE)
        machine_shop = simpy.FilterStore(env, capacity= len(Parameters.MACHINE_PROCESSING_MEAN))
        machine_shop.items = machines
        testing_station = simpy.Resource(env, capacity=1)
        env.process(job_source(env, interval= 1., machines= machine_shop, buffer= buffer, testing_station= testing_station))
        env.run(until= Parameters.SIM_TIME)
        throughputs.append(drain.count)
        cycle_times.append(sum(cycle_time) / len(cycle_time))

    with open(log_dir+time.strftime('%H_%M_%S', time.localtime())+'.txt', "w") as log_file:
        for line in log: log_file.write(str(line) + '\n')

    print(f"mean throughput: {sum(throughputs) / len(throughputs)}")
    print(f"mean cycle time: {sum(cycle_times) / len(cycle_times)}")