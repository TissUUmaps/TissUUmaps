import { serveHierarchicalTable } from "../worker/serveHierarchicalTable";
import { HDF5Store } from "./HDF5Store";

serveHierarchicalTable((source) => HDF5Store.open(source));
