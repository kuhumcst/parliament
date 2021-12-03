(ns dk.cst.parliament.cstlemma
  (:require [clojure.java.shell :refer [sh]]
            [clojure.string :as str])
  (:import [java.io File]))

(def cstlemma-bin
  "/Users/rqf595/Code/cstlemma/cstlemma")

(def flexrules-file
  "/Users/rqf595/Code/tf-idf/resources/flexrules")

(defn cstlemma
  "Return all pairs of [word lemma] for `s` using the cstlemma executable."
  [s]
  (let [file (doto (File/createTempFile "cstlemma" nil)
               (spit s))]
    (-> (sh cstlemma-bin "-L" "-f" flexrules-file "-i" (.getPath file))
        :out
        (str/split #"\n")
        (->> (map #(str/split % #"\t"))
             (map (fn [[k v]]
                    (let [[_ alt] (str/split v #"\|")]
                      [k (or alt v)])))))))

(defn lemmatize
  "Lemmatise `s` using cstlemma."
  [s]
  (str/join " " (map second (cstlemma s))))

(comment
  (cstlemma "Han tog statsborgerskabsprøven efter han var kommet til Danmark.")
  (lemmatize "Han tog statsborgerskabsprøven efter han var kommet til Danmark.")
  #_.)