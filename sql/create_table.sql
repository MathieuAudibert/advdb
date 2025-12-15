CREATE SEQUENCE logs_seq START WITH 1 INCREMENT BY 1 CACHE 20;
CREATE SEQUENCE company_seq START WITH 1 INCREMENT BY 1 CACHE 20;
CREATE SEQUENCE iatype_seq START WITH 1 INCREMENT BY 1 CACHE 20;
CREATE SEQUENCE cfg_seq START WITH 1 INCREMENT BY 1 CACHE 20;
CREATE SEQUENCE specs_seq START WITH 1 INCREMENT BY 1 CACHE 20;
CREATE SEQUENCE iagen_seq START WITH 1 INCREMENT BY 1 CACHE 20;

CREATE TABLE logs_ia (
	id NUMBER DEFAULT logs_seq.nextval PRIMARY KEY,
	type VARCHAR(50) NOT NULL,
	date_creation DATE NOT NULL,
	state VARCHAR(20));

CREATE TABLE Company (
	id NUMBER DEFAULT company_seq.nextval PRIMARY KEY,
	name VARCHAR(255) NOT NULL);

CREATE TABLE IA_Type (
	id NUMBER DEFAULT iatype_seq.nextval PRIMARY KEY,
	category VARCHAR(255) NOT NULL,
	modality VARCHAR(150));

CREATE TABLE Config (
	id NUMBER DEFAULT cfg_seq.nextval PRIMARY KEY,
	api_available NUMBER(1),
	open_source NUMBER(1));

CREATE TABLE Specs (
	id NUMBER DEFAULT specs_seq.nextval PRIMARY KEY,
	mod_text NUMBER(1),
	mod_image NUMBER(1),
	mod_video NUMBER(1),
	mod_audio NUMBER(1),
	mod_code NUMBER(1),
	mod_design NUMBER(1),
	mod_infra NUMBER(1),
	mod_productivity NUMBER(1),
	mod_safety NUMBER(1),
	mod_multimodal NUMBER(1),
	modality_count NUMBER(1));

CREATE TABLE IA_Gen (
	id NUMBER DEFAULT iagen_seq.nextval PRIMARY KEY,
	name VARCHAR(255) NOT NULL,
	website VARCHAR(255),
	release_year DATE,
	fk_specs INT,
	fk_company INT,
	fk_iatype INT,
	fk_cfg INT, 
	FOREIGN KEY (fk_specs) REFERENCES Specs(id),
	FOREIGN KEY (fk_company) REFERENCES Company(id),
	FOREIGN KEY (fk_iatype) REFERENCES IA_Type(id),
	FOREIGN KEY (fk_cfg) REFERENCES Config(id));
