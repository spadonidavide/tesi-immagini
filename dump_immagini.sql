--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: immagini; Type: TABLE; Schema: public; Owner: postgres; Tablespace: 
--

CREATE TABLE immagini (
    id_immagini bigint NOT NULL,
    nome character varying(100)
);


ALTER TABLE public.immagini OWNER TO postgres;

--
-- Name: immagini_id_immagini_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE immagini_id_immagini_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.immagini_id_immagini_seq OWNER TO postgres;

--
-- Name: immagini_id_immagini_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE immagini_id_immagini_seq OWNED BY immagini.id_immagini;


--
-- Name: linee; Type: TABLE; Schema: public; Owner: postgres; Tablespace: 
--

CREATE TABLE linee (
    id_linee bigint NOT NULL,
    l1 integer,
    l2 integer,
    l3 integer,
    l4 integer,
    peso real
);


ALTER TABLE public.linee OWNER TO postgres;

--
-- Name: linee_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE linee_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.linee_id_seq OWNER TO postgres;

--
-- Name: linee_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE linee_id_seq OWNED BY linee.id_linee;


--
-- Name: possiede; Type: TABLE; Schema: public; Owner: postgres; Tablespace: 
--

CREATE TABLE possiede (
    id_possiede bigint NOT NULL,
    fk_id_immagini integer,
    fk_id_linee integer
);


ALTER TABLE public.possiede OWNER TO postgres;

--
-- Name: possiede_id_possiede_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE possiede_id_possiede_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.possiede_id_possiede_seq OWNER TO postgres;

--
-- Name: possiede_id_possiede_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE possiede_id_possiede_seq OWNED BY possiede.id_possiede;


--
-- Name: punti; Type: TABLE; Schema: public; Owner: postgres; Tablespace: 
--

CREATE TABLE punti (
    id_punti bigint NOT NULL,
    r integer,
    g integer,
    b integer
);


ALTER TABLE public.punti OWNER TO postgres;

--
-- Name: punti_id_punti_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE punti_id_punti_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.punti_id_punti_seq OWNER TO postgres;

--
-- Name: punti_id_punti_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE punti_id_punti_seq OWNED BY punti.id_punti;


--
-- Name: id_immagini; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY immagini ALTER COLUMN id_immagini SET DEFAULT nextval('immagini_id_immagini_seq'::regclass);


--
-- Name: id_linee; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY linee ALTER COLUMN id_linee SET DEFAULT nextval('linee_id_seq'::regclass);


--
-- Name: id_possiede; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY possiede ALTER COLUMN id_possiede SET DEFAULT nextval('possiede_id_possiede_seq'::regclass);


--
-- Name: id_punti; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY punti ALTER COLUMN id_punti SET DEFAULT nextval('punti_id_punti_seq'::regclass);


--
-- Data for Name: immagini; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY immagini (id_immagini, nome) FROM stdin;
\.


--
-- Name: immagini_id_immagini_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('immagini_id_immagini_seq', 39928, true);


--
-- Data for Name: linee; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY linee (id_linee, l1, l2, l3, l4, peso) FROM stdin;
\.


--
-- Name: linee_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('linee_id_seq', 12925035, true);


--
-- Data for Name: possiede; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY possiede (id_possiede, fk_id_immagini, fk_id_linee) FROM stdin;
\.


--
-- Name: possiede_id_possiede_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('possiede_id_possiede_seq', 671867, true);


--
-- Data for Name: punti; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY punti (id_punti, r, g, b) FROM stdin;
\.


--
-- Name: punti_id_punti_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('punti_id_punti_seq', 511914, true);


--
-- Name: immagini_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY immagini
    ADD CONSTRAINT immagini_pkey PRIMARY KEY (id_immagini);


--
-- Name: linee_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY linee
    ADD CONSTRAINT linee_pkey PRIMARY KEY (id_linee);


--
-- Name: possiede_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY possiede
    ADD CONSTRAINT possiede_pkey PRIMARY KEY (id_possiede);


--
-- Name: punti_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres; Tablespace: 
--

ALTER TABLE ONLY punti
    ADD CONSTRAINT punti_pkey PRIMARY KEY (id_punti);


--
-- Name: linee_l1_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY linee
    ADD CONSTRAINT linee_l1_fkey FOREIGN KEY (l1) REFERENCES punti(id_punti);


--
-- Name: linee_l2_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY linee
    ADD CONSTRAINT linee_l2_fkey FOREIGN KEY (l2) REFERENCES punti(id_punti);


--
-- Name: linee_l3_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY linee
    ADD CONSTRAINT linee_l3_fkey FOREIGN KEY (l3) REFERENCES punti(id_punti);


--
-- Name: linee_l4_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY linee
    ADD CONSTRAINT linee_l4_fkey FOREIGN KEY (l4) REFERENCES punti(id_punti);


--
-- Name: possiede_fk_id_immagini_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY possiede
    ADD CONSTRAINT possiede_fk_id_immagini_fkey FOREIGN KEY (fk_id_immagini) REFERENCES immagini(id_immagini);


--
-- Name: possiede_fk_id_linee_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY possiede
    ADD CONSTRAINT possiede_fk_id_linee_fkey FOREIGN KEY (fk_id_linee) REFERENCES linee(id_linee);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- Name: immagini; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON TABLE immagini FROM PUBLIC;
REVOKE ALL ON TABLE immagini FROM postgres;
GRANT ALL ON TABLE immagini TO postgres;
GRANT ALL ON TABLE immagini TO davide;


--
-- Name: immagini_id_immagini_seq; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON SEQUENCE immagini_id_immagini_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE immagini_id_immagini_seq FROM postgres;
GRANT ALL ON SEQUENCE immagini_id_immagini_seq TO postgres;
GRANT SELECT,UPDATE ON SEQUENCE immagini_id_immagini_seq TO davide;


--
-- Name: linee; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON TABLE linee FROM PUBLIC;
REVOKE ALL ON TABLE linee FROM postgres;
GRANT ALL ON TABLE linee TO postgres;
GRANT ALL ON TABLE linee TO davide;


--
-- Name: linee_id_seq; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON SEQUENCE linee_id_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE linee_id_seq FROM postgres;
GRANT ALL ON SEQUENCE linee_id_seq TO postgres;
GRANT SELECT,UPDATE ON SEQUENCE linee_id_seq TO davide;


--
-- Name: possiede; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON TABLE possiede FROM PUBLIC;
REVOKE ALL ON TABLE possiede FROM postgres;
GRANT ALL ON TABLE possiede TO postgres;
GRANT ALL ON TABLE possiede TO davide;


--
-- Name: possiede_id_possiede_seq; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON SEQUENCE possiede_id_possiede_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE possiede_id_possiede_seq FROM postgres;
GRANT ALL ON SEQUENCE possiede_id_possiede_seq TO postgres;
GRANT SELECT,UPDATE ON SEQUENCE possiede_id_possiede_seq TO davide;


--
-- Name: punti; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON TABLE punti FROM PUBLIC;
REVOKE ALL ON TABLE punti FROM postgres;
GRANT ALL ON TABLE punti TO postgres;
GRANT ALL ON TABLE punti TO davide;


--
-- Name: punti_id_punti_seq; Type: ACL; Schema: public; Owner: postgres
--

REVOKE ALL ON SEQUENCE punti_id_punti_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE punti_id_punti_seq FROM postgres;
GRANT ALL ON SEQUENCE punti_id_punti_seq TO postgres;
GRANT SELECT,UPDATE ON SEQUENCE punti_id_punti_seq TO davide;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: davide
--

ALTER DEFAULT PRIVILEGES FOR ROLE davide IN SCHEMA public REVOKE ALL ON TABLES  FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE davide IN SCHEMA public REVOKE ALL ON TABLES  FROM davide;
ALTER DEFAULT PRIVILEGES FOR ROLE davide IN SCHEMA public GRANT SELECT,INSERT,DELETE,UPDATE ON TABLES  TO davide;


--
-- PostgreSQL database dump complete
--

