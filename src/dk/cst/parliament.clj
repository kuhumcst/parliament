(ns dk.cst.parliament
  "Experiments on the Danish parliament dataset."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.data.csv :as csv]
            [clojure.data.json :as json]
            [clojure.set :as set]
            [tech.v3.dataset :as ds]
            [libpython-clj2.python :as py]
            [scicloj.sklearn-clj :as sklearn]
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

(defonce bad-dates
  (atom #{}))

;; Some of the birthdates in the dataset are provided as yyyy-dd-MM.
(defn- fix-danish-date-format
  [s]
  (when s
    (if (re-matches #"0\d|1(0|1|2)" (subs s 5 7))
      s
      (do
        (swap! bad-dates conj s)
        (str (subs s 0 5) (subs s 8 10) (subs s 4 7))))))

(defn load-csv-maps
  [dir-path & options]
  (let [root-dir   (io/file dir-path)
        directory? (fn [file] (.isDirectory file))
        files      (remove directory? (file-seq root-dir))
        read-rows  (comp #(apply csv/read-csv % options) io/reader)
        columns    (->> (mapcat read-rows files)
                        (take 1)
                        (first)
                        (map (comp keyword #(str/replace % #"\s" "-"))))
        ->map      (comp #(update % :Birth fix-danish-date-format)
                         (partial zipmap columns))]
    (mapcat (comp (partial map ->map)
                  rest
                  read-rows)
            files)))

(def parse-methods
  {:Birth      [:int64 (fn [s] (parse-long (subs s 0 4)))]
   #_#_:Start-time :local-time
   #_#_:Date       [:local-date "yyyy-MM-dd"]
   #_#_:End-time   :local-time
   :Time       :int64
   :Age        :int64})

(defn ->ds
  [dir]
  (-> (load-csv-maps dir :separator \tab)
      (ds/->dataset {:parser-fn parse-methods})
      (vary-meta assoc :print-column-max-width 20)))

(defn ->vocab-map
  "Make a bridged Python dict of the vocabulary of `documents` limited by `n`;
  follows format prescribed by e.g. sklearn CountVectorizer or TfidfVectorizer."
  [documents n]
  (-> (tf-idf/vocab documents n)
      (zipmap (range))))

(comment

  (spit "resources/parliament/vocabulary.json" (json/write-str (->vocab-map documents 3)))


  (def vocab-dict
    (->vocab-dict documents 3))

  (sklearn/fit (ds/select ds [:Lemma] :all) "sklearn.feature_extraction.text" "CountVectorizer")

  (def fitted
    (-> (ds/select ds [:Lemma] :all)
        (ds/filter (comp some? :Lemma))
        (sklearn/fit-transform "sklearn.feature_extraction.text" "CountVectorizer" {:vocabulary vocab-dict})))

  (py/py. fitted toarray)

  (def ds
    (->ds (io/resource "parliament/lemmatized")))

  (ds/write! ds "resources/parliament/data.csv")

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
