Parliament corpus experiments
=============================

Setup
-----

### Python dependencies
This project is written in Clojure, but it uses does use some Python libraries which need to be installed before attempting to run the code:

```
# your mileage may vary
pip3 intall -U scikit-learn
```

(and of course Python itself needs to be installed too)

The `:development` alias should always be included as this sets up the Python translation layer in the default `user.clj` file (`user` is the default namespace used when launching a fresh REPL).

### JDK 17
When running on JDK version 17 and above the `:jdk-17` alias needs to be included.
