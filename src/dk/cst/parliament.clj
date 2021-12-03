(ns dk.cst.parliament
  "Experiments on the Danish parliament dataset."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.data.csv :as csv]
            [clojure.set :as set]
            [dk.cst.tf-idf :as tf-idf]))

(def parliament-stopwords
  #{"ordfører"
    "ordføreren"
    "minister"
    "ministeren"
    "spørger"
    "spørgeren"
    "hr"
    "fru"
    "værsgo"
    "tak"
    "kort"
    "korte"
    "bemærkning"
    "bemærkninger"})

(def danish-stopwords
  "Source: https://github.com/stopwords-iso/stopwords-da"
  (-> (io/resource "stopwords-da.txt")
      (slurp)
      (str/split #"\n")
      (set)))

(def stopwords
  (set/union parliament-stopwords danish-stopwords))

(def danish-tokenizer-xf
  (tf-idf/->tokenizer-xf
    :postprocess (partial remove stopwords)))

(defn load-csv
  "Load the files in `dir-path` as CSV; takes same `options` as 'csv/read-csv'"
  [dir-path & options]
  (let [root-dir   (io/file dir-path)
        directory? (fn [file] (.isDirectory file))
        files      (remove directory? (file-seq root-dir))]
    (mapcat (comp rest #(apply csv/read-csv % options) io/reader) files)))

(comment
  ;; Parliament corpus data (lazy-loaded)
  (def documents-raw
    (map last (load-csv (io/resource "parliament/raw") :separator \tab)))

  ;; Lemmatized corpus data
  (def documents
    (map last (load-csv (io/resource "parliament/lemmatized") :separator \tab)))

  (def df-result
    (tf-idf/df documents))

  ;; full vocab (raw)
  (count (tf-idf/vocab documents-raw))                      ; 212878

  ;; full vocab (lemmatized)
  (count (tf-idf/vocab df-result))                          ; 169583

  ;; ... with rare terms removed (lemmatized)
  (count (tf-idf/vocab df-result 1))                        ; 86443
  (count (tf-idf/vocab df-result 2))                        ; 64935
  (count (tf-idf/vocab df-result 3))                        ; 54084
  (count (tf-idf/vocab df-result 4))                        ; 47301
  (count (tf-idf/vocab df-result 5))                        ; 42510

  (def tf-idf-results
    (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
      (tf-idf/tf-idf (take 100 documents))))

  ;; Warning: heavy computation
  (def top-cut-off-keywords
    (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
      (tf-idf/top-n-terms 3 (tf-idf/tf-idf documents))))

  ;; TODO: more than 100k results - need a better keyword ranking algorithm
  (count top-cut-off-keywords)

  ;; TODO: remove names during pre/post-processing?
  ;; Warning: heavy computation
  (def top-200-keywords
    (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
      (take 200 (tf-idf/top-sum-terms (tf-idf/tf-idf documents)))))

  (def top-200-keywords
    (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
      (take 200 (tf-idf/top-max-terms (tf-idf/tf-idf documents)))))

  ;; Less heavy computation, for testing
  (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
    (tf-idf/top-n-terms 3 (take 1000 tf-idf-results))) |

  ;; Check that stopwords are removed from results
  (set/difference (tf-idf/top-n-terms 3 (take 1000 (tf-idf/tf-idf documents)))
                  (binding [tf-idf/*tokenizer-xf* danish-tokenizer-xf]
                    (tf-idf/top-n-terms 3 (take 1000 (tf-idf/tf-idf documents)))))
  #_.)
