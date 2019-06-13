import logging
import mdtraj as md
import numpy as np
import yank

from simtk import openmm, unit
from simtk.openmm import app
from mdtraj.reporters import NetCDFReporter

import openmmtools
from simtk.openmm import XmlSerializer
from openmmtools import states, mcmc
from openmmtools.states import GlobalParameterState
from openmmtools.multistate import SAMSSampler, MultiStateReporter
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion, AlchemicalState
import itertools

class MyComposableState(GlobalParameterState):
     lambda_sterics = GlobalParameterState.GlobalParameter('lambda_sterics', standard_value=1.0)
     lambda_electrostatics = GlobalParameterState.GlobalParameter('lambda_electrostatics', standard_value=1.0)
     lambda_restraints = GlobalParameterState.GlobalParameter('lambda_restraints', standard_value=12/160)
     K_parallel = GlobalParameterState.GlobalParameter('K_parallel',
                                                     standard_value=6000*unit.kilojoules_per_mole/unit.nanometer**2)
     Kmax = GlobalParameterState.GlobalParameter('Kmax',
                                                 standard_value=100*unit.kilojoules_per_mole/unit.nanometer**2)
     Kmin = GlobalParameterState.GlobalParameter('Kmin',
                                                 standard_value=10*unit.kilojoules_per_mole/unit.nanometer**2)

yank.utils.config_root_logger(verbose=True, log_file_path=None)
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
cv_logger = logging.getLogger(__name__)
cv_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('cv.log')
cv_logger.addHandler(handler)

from yank.experiment import ExperimentBuilder
platform = ExperimentBuilder._configure_platform('CUDA', 'mixed')

try:
    openmmtools.cache.global_context_cache.platform = platform
except RuntimeError:
    openmmtools.cache.global_context_cache.empty()
    openmmtools.cache.global_context_cache.platform = platform

# Topology
pdb = app.PDBFile('../../../smd/porin-ligand.pdb')

with open('../../../smd/openmm_system_alch.xml', 'r') as infile:
    openmm_system = XmlSerializer.deserialize(infile.read())


topology = md.Topology.from_openmm(pdb.topology)
ligand_atoms = topology.select('(resname CM7)')

factory = AbsoluteAlchemicalFactory(consistent_exceptions=False, disable_alchemical_dispersion_correction=True)
alchemical_region = AlchemicalRegion(alchemical_atoms=ligand_atoms)
alchemical_system = factory.create_alchemical_system(openmm_system, alchemical_region)

thermodynamic_state_ref = states.ThermodynamicState(system=alchemical_system,
                                                    temperature=310*unit.kelvin,
                                                    pressure=1.0*unit.atmospheres)

sampler_state = states.SamplerState(positions=pdb.positions, box_vectors=pdb.topology.getPeriodicBoxVectors())
nstates = 160
#lambda_sterics= [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.59, 0.54, 0.47, 0.41, 0.36, 0.31, 0.27, 0.25, 0.225, 0.20, 0.18, 0.15, 0.13,
#0.11, 0.06, 0.03, 0.00]
#lambda_sterics =[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.59, 0.54, 0.47,
#0.41, 0.36, 0.31, 0.27, 0.25, 0.225, 0.20, 0.18, 0.15, 0.13, 0.11, 0.06, 0.03, 0.00]
#lambda_electrostatics=[1.0, 0.9, 0.79, 0.69, 0.58, 0.46, 0.32, 0.18, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

lambda_electrostatics=[1.0, 0.95, 0.9, 0.85, 0.8]
lambda_sterics=[1.0, 1.0, 1.0, 1.0, 1.0]
lambda_restraints=[ i/nstates for i in range(12, 112)]

K_parallel=[6000*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(len(lambda_restraints))]
Kmax=[100*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(len(lambda_restraints))]
Kmin=[10*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(len(lambda_restraints))]

lambda_sterics_2d=lambda_sterics*len(lambda_restraints)
lambda_electrostatics_2d=lambda_electrostatics*len(lambda_restraints)
lambda_restraints_2d = list(itertools.chain.from_iterable(itertools.repeat(x, len(lambda_electrostatics)) for x in lambda_restraints))
K_parallel_2d = list(itertools.chain.from_iterable(itertools.repeat(x, len(lambda_electrostatics)) for x in K_parallel))
K_orthogonal_max_2d = list(itertools.chain.from_iterable(itertools.repeat(x, len(lambda_electrostatics)) for x in Kmax))
K_orthogonal_min_2d = list(itertools.chain.from_iterable(itertools.repeat(x, len(lambda_electrostatics)) for x in Kmin))

protocol = {'lambda_sterics': lambda_sterics_2d,
            'lambda_electrostatics': lambda_electrostatics_2d,
            'lambda_restraints': lambda_restraints_2d,
            'K_parallel': K_parallel_2d,
            'Kmax': K_orthogonal_max_2d,
            'Kmin': K_orthogonal_min_2d}


my_composable_state = MyComposableState.from_system(alchemical_system)
compound_states = states.create_thermodynamic_state_protocol(thermodynamic_state_ref,
                                                             protocol=protocol,
                                                             composable_states=[my_composable_state])

move = mcmc.LangevinDynamicsMove(timestep=4*unit.femtosecond,
                                 collision_rate= 1.0/unit.picoseconds,
                                 n_steps=500,
                                 reassign_velocities=False)
simulation = SAMSSampler(mcmc_moves=move,
                         minimum_round_trips=1,
                         histogram_flatness=0.3,
                         number_of_iterations=1,
                         online_analysis_interval=None,
                         beta_factor=0.6)

analysis_particle_indices = topology.select('(protein and mass > 3.0) or (resname CM7 and mass > 3.0)')
reporter = MultiStateReporter('alchemical_test.nc', checkpoint_interval=50, analysis_particle_indices=analysis_particle_indices)
simulation.create(thermodynamic_states=compound_states,
                  sampler_states=[sampler_state],
                  storage=reporter)
simulation.extend(n_iterations=1)
for step in range(100000):
    ts = simulation._thermodynamic_states[simulation._replica_thermodynamic_states[0]]
    context, _ = openmmtools.cache.global_context_cache.get_context(ts)
    context.setParameter('report', 1.0)
    cv_logger.debug('{}, {}, {}, {}, {}'.format(context.getParameter('lambda_restraints'),
                                 context.getState(getEnergy=True, groups=2**30).getPotentialEnergy()._value,
                                 context.getState(getEnergy=True,groups=2**31).getPotentialEnergy()._value,
                                 context.getParameter('lambda_sterics'),
                                 context.getParameter('lambda_electrostatics')))
    context.setParameter('report', 0.0)
    simulation.extend(n_iterations=1)
