"""
Export the human part of the Reactome graph database (Neo4j 4.x) to CSV files.

Usage (Windows, with Neo4j running and the Reactome graph.db loaded):
    pip install neo4j
    python export_reactome_csv.py
It asks for the Neo4j password and writes the CSV files into .\reactome_csv\

Each query only reads (MATCH ... RETURN). Nothing in the database is changed.
"""
import csv
import getpass
import os
import time

from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
DATABASE = os.environ.get("NEO4J_DATABASE") or None  # None = the default database (graph.db)
OUT = "reactome_csv"
HUMAN = "Homo sapiens"

# name -> (Cypher query, column names). All queries are restricted to human.
QUERIES = {
    # ---------------- nodes ----------------
    "nodes_pathway": ("""
        MATCH (p:Pathway {speciesName: $sp})
        RETURN p.stId, p.displayName, p.schemaClass, p.isInDisease, p.hasDiagram
        """, ["pathway_id", "name", "schema_class", "is_in_disease", "has_diagram"]),

    "nodes_reaction": ("""
        MATCH (r:ReactionLikeEvent {speciesName: $sp})
        RETURN r.stId, r.displayName, r.schemaClass, r.isInDisease
        """, ["reaction_id", "name", "schema_class", "is_in_disease"]),

    "nodes_physical_entity": ("""
        MATCH (pe:PhysicalEntity {speciesName: $sp})
        RETURN pe.stId, pe.displayName, pe.schemaClass
        """, ["entity_id", "name", "schema_class"]),

    "nodes_small_molecule": ("""
        MATCH (se:SimpleEntity)-[:referenceEntity]->(rm:ReferenceMolecule)
        RETURN se.stId, se.displayName, rm.identifier, rm.databaseName
        """, ["entity_id", "name", "reference_id", "reference_db"]),

    "nodes_gene_product": ("""
        MATCH (ewas:EntityWithAccessionedSequence {speciesName: $sp})-[:referenceEntity]->(re:ReferenceEntity)
        RETURN ewas.stId, ewas.displayName, re.identifier, re.databaseName, re.geneName
        """, ["entity_id", "name", "reference_id", "reference_db", "gene_names"]),

    # ---------------- edges ----------------
    "edges_pathway_has_event": ("""
        MATCH (p:Pathway {speciesName: $sp})-[:hasEvent]->(e:Event)
        RETURN p.stId, e.stId, e.schemaClass
        """, ["pathway_id", "child_event_id", "child_schema_class"]),

    "edges_reaction_input_output": ("""
        MATCH (r:ReactionLikeEvent {speciesName: $sp})-[rel:input|output]->(pe:PhysicalEntity)
        RETURN r.stId, type(rel), pe.stId, rel.stoichiometry
        """, ["reaction_id", "role", "entity_id", "stoichiometry"]),

    "edges_reaction_catalyst": ("""
        MATCH (r:ReactionLikeEvent {speciesName: $sp})-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(pe:PhysicalEntity)
        RETURN r.stId, 'catalyst', pe.stId
        """, ["reaction_id", "role", "entity_id"]),

    "edges_reaction_regulator": ("""
        MATCH (r:ReactionLikeEvent {speciesName: $sp})-[:regulatedBy]->(reg:Regulation)-[:regulator]->(pe:PhysicalEntity)
        RETURN r.stId, reg.schemaClass, pe.stId
        """, ["reaction_id", "regulation_type", "entity_id"]),

    "edges_complex_component": ("""
        MATCH (c:Complex {speciesName: $sp})-[rel:hasComponent]->(pe:PhysicalEntity)
        RETURN c.stId, pe.stId, rel.stoichiometry
        """, ["complex_id", "component_id", "stoichiometry"]),

    "edges_set_member": ("""
        MATCH (s:EntitySet {speciesName: $sp})-[rel:hasMember|hasCandidate]->(pe:PhysicalEntity)
        RETURN s.stId, type(rel), pe.stId
        """, ["set_id", "relation", "member_id"]),
}


def to_cell(v):
    if isinstance(v, list):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def main():
    password = os.environ.get("NEO4J_PASSWORD") or getpass.getpass(f"Neo4j password for {USER}@{URI}: ")
    os.makedirs(OUT, exist_ok=True)
    driver = GraphDatabase.driver(URI, auth=(USER, password))
    with driver.session(database=DATABASE) as session:
        # quick sanity check before the full export
        n = session.run("MATCH (p:Pathway {speciesName: $sp}) RETURN count(p) AS n", sp=HUMAN).single()["n"]
        print(f"human pathways in the database: {n}")
        for name, (cypher, header) in QUERIES.items():
            t = time.time()
            path = os.path.join(OUT, f"{name}.csv")
            rows = 0
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                for rec in session.run(cypher, sp=HUMAN):
                    w.writerow([to_cell(v) for v in rec.values()])
                    rows += 1
            print(f"{name:32s} {rows:>9,d} rows  {time.time() - t:6.1f}s  -> {path}")
    driver.close()
    print("done")


if __name__ == "__main__":
    main()
