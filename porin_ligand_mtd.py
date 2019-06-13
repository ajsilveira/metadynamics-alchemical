from simtk import openmm, unit
from simtk.openmm import app
import metadynamics as mtd
import numpy as np
import mdtraj as md
from mdtraj.reporters import DCDReporter
from mdtraj.reporters import NetCDFReporter
from simtk.openmm import XmlSerializer

pressure = 1.0 * unit.atmospheres
temperature = 310.0 * unit.kelvin
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
pdb = app.PDBFile('../../smd/porin-ligand.pdb')

with open('../../smd/openmm_system_meta.xml', 'r') as infile:
  openmm_system = XmlSerializer.deserialize(infile.read())

# openmm_system already contains a custom force r_parallel that it is used to access
# the values of r_parallel and r_orthogonal. This force was created as follows:

# common = 'r_parallel = r*cos(theta);'
# common += 'r_orthogonal = r*sin(theta);'
# common += 'r = distance(g1,g2);'
# common += 'theta = angle(g1,g2,g3);'
# r_parallel = openmm.CustomCentroidBondForce(3, 'report*r_parallel;' + common )
# r_orthogonal = openmm.CustomCentroidBondForce(3, 'report*r_orthogonal;' + common )
# r_parallel.setForceGroup(29)
# r_orthogonal.setForceGroup(30)
# for cv in [r_parallel, r_orthogonal]:
#         cv.addGlobalParameter('report', 0.0)
#         cv.addGroup([int(index) for index in ligand_atoms])
#         cv.addGroup([int(index) for index in bottom_atoms], [1.0, 1.0, 1.0])
#         cv.addGroup([int(index) for index in top_atoms], [1.0, 1.0, 1.0])
#         cv.addBond([0,1,2], [])
#         system.addForce(cv)


md_topology = md.Topology.from_openmm(pdb.topology)
topology = pdb.topology

bottom_atoms = [1396, 5018, 3076]
top_atoms = [6241, 3325, 1626]
selection = '(resname CM7) and (mass > 3.0)'
print('Determining ligand atoms using "{}"...'.format(selection))
ligand_atoms = md_topology.select(selection)

r_parallel = openmm.CustomCentroidBondForce(3, '0')
r_parallel.addGroup([int(index) for index in ligand_atoms])
r_parallel.addGroup([int(index) for index in bottom_atoms], [1.0, 1.0, 1.0])
r_parallel.addGroup([int(index) for index in top_atoms], [1.0, 1.0, 1.0])
r_parallel.addBond([0,1,2], [])

bv = mtd.BiasVariable(r_parallel_bias, -2.0, 3.0, 0.1, False)
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
meta = mtd.Metadynamics(openmm_system, [bv], r_parallel, temperature, 10, 5*unit.kilojoules_per_mole, 1000, saveFrequency=20000, biasDir='/data/chodera/silveira/OprD/gromacs/metadynamics/')

simulation = app.Simulation(pdb.topology, openmm_system, integrator)
simulation.context.setPositions(pdb.positions)
analysis_particle_indices = md_topology.select('(protein and mass > 3.0) or (resname CM7 and mass > 3.0)')

mdtraj_reporter = DCDReporter('trajectory.dcd', 50000, atomSubset=analysis_particle_indices)
simulation.reporters.append(mdtraj_reporter)
meta.step(simulation, 75000000)
