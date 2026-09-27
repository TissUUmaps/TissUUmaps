import { serveHierarchicalTable } from "../workers/serveHierarchicalTable";
import { HDF5Store } from "./HDF5Store";

serveHierarchicalTable((source) => HDF5Store.open(source));
