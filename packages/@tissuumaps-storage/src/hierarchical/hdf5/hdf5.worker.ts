import { startHierarchicalTableServer } from "../workers/startHierarchicalTableServer";
import { HDF5Store } from "./HDF5Store";

startHierarchicalTableServer((source) => HDF5Store.open(source));
