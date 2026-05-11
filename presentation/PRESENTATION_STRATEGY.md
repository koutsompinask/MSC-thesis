# Στρατηγική Ζωντανής Παρουσίασης Διπλωματικής

## Κεντρικός Στόχος

Ο στόχος δεν είναι να διαβάσεις τη διπλωματική. Ο στόχος είναι να δείξεις στην επιτροπή ότι:

- Καταλαβαίνεις το πρόβλημα της ανίχνευσης απάτης.
- Ο πειραματικός σχεδιασμός σου είναι προσεκτικός και αποφεύγει data leakage.
- Τα αποτελέσματα απαντούν καθαρά στα τέσσερα ερευνητικά ερωτήματα.
- Καταλαβαίνεις τους περιορισμούς και τα επιχειρησιακά trade-offs.
- Μπορείς να εξηγήσεις γιατί επέλεξες chronological split, ROC-AUC, downsampling, SHAP και threshold tuning.

Κεντρική πρόταση της παρουσίασης:

> Η διπλωματική αξιολογεί μοντέλα gradient boosting για ανίχνευση απάτης στο IEEE-CIS dataset, χρησιμοποιώντας χρονικά συνεπή πειραματικό σχεδιασμό ώστε να μελετήσει τη διακριτική ικανότητα των μοντέλων, την ανισορροπία κλάσεων, τη μείωση χαρακτηριστικών και τα επιχειρησιακά trade-offs του decision threshold.

## Ευθυγράμμιση με τις Οδηγίες Παρουσίασης

Οι οδηγίες ζητούν μια καθαρή ιστορία περίπου 15 λεπτών ή περισσότερο, συνήθως με 10-20 διαφάνειες, που καλύπτει πρόβλημα, μεθοδολογία, ευρήματα, συμπεράσματα και άμυνα. Η παρουσίαση έχει 21 διαφάνειες, άρα είναι λίγο πάνω από τη σύσταση, αλλά είναι αποδεκτή αν κάποιες διαφάνειες παρουσιαστούν γρήγορα.

Ισχυρή ευθυγράμμιση:

- Υπάρχουν τίτλος, roadmap, πλαίσιο προβλήματος, ερευνητικά ερωτήματα, βιβλιογραφία, μεθοδολογία, dataset, αποτελέσματα, σύνθεση, συμπεράσματα και τελική διαφάνεια.
- Τα αποτελέσματα ακολουθούν τη δομή της διπλωματικής: baseline, downsampling, SHAP/reduced features, threshold tuning, synthesis.
- Οι διαφάνειες δεν αντιγράφουν μεγάλες παραγράφους από τη διπλωματική.
- Τα ερευνητικά ερωτήματα είναι αριθμημένα και συνδέονται άμεσα με τα πειράματα.

Σημεία που καλύπτεις προφορικά:

- Δεν υπάρχει ξεχωριστή slide βιβλιογραφίας. Επειδή δεν αναφέρεις συγκεκριμένους συγγραφείς στις διαφάνειες, αυτό είναι αποδεκτό. Αν ρωτηθείς, παραπέμπεις στο κεφάλαιο βιβλιογραφίας της διπλωματικής.
- Οι περιορισμοί πρέπει να ειπωθούν καθαρά στη διαφάνεια 19.
- Επειδή οι διαφάνειες είναι 21, κράτα τη διαφάνεια 2, τη διαφάνεια 5, τη διαφάνεια 17 και τη διαφάνεια 21 σύντομες.

## Πλάνο Χρόνου

Στόχευσε σε 16-18 λεπτά συνολικά, συν τις ερωτήσεις.

- Διαφάνειες 1-2: 1 λεπτό.
- Διαφάνειες 3-5: 2 λεπτά.
- Διαφάνειες 6-11: 5 λεπτά.
- Διαφάνειες 12-18: 6-7 λεπτά.
- Διαφάνεια 19: 1.5 λεπτό.
- Διαφάνεια 20 live demo: 1 λεπτό.
- Διαφάνεια 21: 10 δευτερόλεπτα.

Αν πιεστείς χρονικά, μείωσε πολύ τη διαφάνεια 5 και πες μόνο μία πρόταση στη διαφάνεια 17.

## Κείμενο Παρουσίασης Ανά Διαφάνεια

### Διαφάνεια 1 - Τίτλος

Πες:

> Καλημέρα σας. Η διπλωματική μου έχει τίτλο "Machine Learning for Fraud Detection in Online Financial Transactions". Η παρουσίαση εστιάζει στο behavioral fraud analytics, χρησιμοποιώντας το IEEE-CIS fraud detection dataset. Το βασικό ερώτημα είναι πόσο αποτελεσματικά μπορούν τα gradient boosting μοντέλα να εντοπίσουν απάτη σε ένα πολύ ανισορροπημένο σύνολο συναλλαγών, και πώς οι επιλογές μοντελοποίησης επηρεάζουν την πρακτική συμπεριφορά του συστήματος.

Μην εξηγήσεις υπερβολικά τη διαφάνεια τίτλου.

### Διαφάνεια 2 - Roadmap

Πες:

> Αρχικά θα παρουσιάσω την πρόκληση της ανίχνευσης απάτης και το σχετικό machine learning πλαίσιο. Στη συνέχεια θα περάσω στα ερευνητικά ερωτήματα, το dataset, τη μεθοδολογία, τα πειράματα, τα αποτελέσματα και τα συμπεράσματα. Στο τέλος θα δείξω ένα σύντομο live demo του pipeline.

Κράτησέ το κάτω από 25 δευτερόλεπτα.

### Διαφάνεια 3 - Το Πρόβλημα

Πες:

> Η βασική δυσκολία είναι ότι η απάτη είναι σπάνια. Στο συγκεκριμένο dataset μόνο περίπου 3.5% των συναλλαγών είναι fraudulent, άρα το accuracy από μόνο του μπορεί να είναι παραπλανητικό. Ένα μοντέλο μπορεί να φαίνεται ακριβές, αλλά να αποτυγχάνει να εντοπίσει τη μειοψηφική κλάση. Η δεύτερη δυσκολία είναι επιχειρησιακή: αν θέλουμε να πιάσουμε περισσότερη απάτη, συνήθως αυξάνονται και τα false positives. Γι' αυτό η διπλωματική εξετάζει και discrimination metrics όπως ROC-AUC, αλλά και threshold-dependent metrics όπως precision και recall.

Φράση άμυνας:

> Το πρόβλημα δεν είναι μόνο η ακρίβεια του μοντέλου. Είναι το trade-off ανάμεσα στο fraud capture και τα false alerts.

### Διαφάνεια 4 - Ερευνητικά Ερωτήματα

Πες:

> Η μελέτη οργανώνεται γύρω από τέσσερα ερευνητικά ερωτήματα. Το πρώτο εξετάζει πόσο καλά τα XGBoost, LightGBM και CatBoost διακρίνουν fraud από legitimate συναλλαγές. Το δεύτερο εξετάζει την επίδραση του majority-class downsampling. Το τρίτο εξετάζει αν ένα μικρότερο feature set, επιλεγμένο με βάση cross-model importance agreement, μπορεί να διατηρήσει την απόδοση. Το τέταρτο εξετάζει πώς η αλλαγή του decision threshold επηρεάζει recall και precision.

Πρόσθεσε:

> Αυτά τα ερωτήματα αντιστοιχούν άμεσα στις πειραματικές διαμορφώσεις που θα δούμε στα αποτελέσματα.

### Διαφάνεια 5 - Βιβλιογραφικό Πλαίσιο

Πες:

> Η βιβλιογραφία περιλαμβάνει supervised models, anomaly detection, deep learning και ensemble methods. Σε αυτή τη διπλωματική εστίασα στα gradient boosting μοντέλα, επειδή το dataset είναι structured tabular transaction data. Σε τέτοια δεδομένα τα gradient boosting models είναι συνήθως πολύ ισχυρά, πρακτικά στην εκπαίδευση, και μπορούν να εξηγηθούν με εργαλεία όπως το SHAP.

Απόφυγε:

- Μην πεις ότι το deep learning είναι γενικά κακό.
- Πες ότι δεν ήταν το κύριο focus για αυτό το dataset και αυτόν τον πειραματικό σχεδιασμό.

### Διαφάνεια 6 - Dataset

Πες:

> Το dataset είναι το IEEE-CIS fraud detection dataset από το Kaggle. Χρησιμοποίησα το labeled training data, με 590,540 συναλλαγές και 434 αρχικές feature columns πριν το feature engineering. Ο στόχος είναι binary: isFraud ίσον 1 για fraudulent και 0 για legitimate συναλλαγές. Το dataset περιλαμβάνει στοιχεία συναλλαγής, κάρτας, email, device, identity και Vesta-engineered features. Είναι κατάλληλο επειδή είναι μεγάλο, δημόσιο, ανωνυμοποιημένο και ρεαλιστικά ανισορροπημένο.

Σημαντική διευκρίνιση:

> Το 434 αναφέρεται στις αρχικές στήλες του dataset. Μετά το feature engineering, ο χώρος εισόδου των μοντέλων έγινε μεγαλύτερος: 748 features πριν τη μείωση.

### Διαφάνεια 7 - Methodology Pipeline

Πες:

> Το pipeline ξεκινά με exploratory data analysis, μετά preprocessing και feature engineering, και στη συνέχεια model training με Optuna tuning και time-series cross-validation. Μετά τα baseline μοντέλα, εξετάζω downsampling, SHAP-based feature reduction και threshold tuning. Η σειρά είναι σημαντική, γιατί η διπλωματική δεν ρωτά μόνο ποιο μοντέλο κερδίζει, αλλά πώς συμπεριφέρεται ολόκληρο το fraud detection pipeline.

Φράση άμυνας:

> Τα πειράματα είναι συγκρίσιμα επειδή χρησιμοποιούν το ίδιο chronological evaluation protocol.

### Διαφάνεια 8 - Chronological Split

Πες:

> Αυτή είναι μία από τις πιο σημαντικές μεθοδολογικές επιλογές. Δεν χρησιμοποίησα random split. Οι συναλλαγές ταξινομήθηκαν με βάση το TransactionDT, το πρώτο 80% χρησιμοποιήθηκε για training, και το τελευταίο 20% κρατήθηκε ως unseen test data. Το hyperparameter tuning έγινε μέσα στο training period, με expanding time-series cross-validation. Αυτό μειώνει τον κίνδυνο να περάσει μελλοντική πληροφορία στο training.

Αν σε πιέσουν:

> Ένα random split μπορεί να δώσει υψηλότερα scores, αλλά είναι λιγότερο ρεαλιστικό για fraud detection, επειδή τα fraud patterns εξελίσσονται χρονικά.

### Διαφάνεια 9 - EDA

Πες:

> Το EDA εξηγεί γιατί χρειάστηκαν οι επόμενες μοντελοποιητικές επιλογές. Η ανισορροπία κλάσεων εξηγεί γιατί δεν βασιζόμαστε στο accuracy. Το μεγάλο ποσοστό missing values δείχνει ότι τα identity και device fields είναι δύσκολα αλλά δυνητικά χρήσιμα, οπότε αφαιρέθηκαν μόνο στήλες με ακραίο missingness. Τα χρονικά και amount-related patterns οδήγησαν στη δημιουργία behavioral aggregate features.

Κεντρικό σημείο:

> Τα engineered features δεν ήταν αυθαίρετα. Προέκυψαν από τα patterns που φάνηκαν στο EDA.

### Διαφάνεια 10 - Feature Engineering

Πες:

> Επειδή το dataset είναι ανωνυμοποιημένο, δεν υπάρχει πραγματικό customer ID. Δημιούργησα ένα simulated user anchor από πεδία όπως card1, addr1 και ένα account-age proxy. Στη συνέχεια δημιούργησα behavioral aggregate features: means, standard deviations, relative deviations, counts και frequencies. Η ιδέα είναι κάθε συναλλαγή να συγκρίνεται όχι μόνο συνολικά, αλλά και σε σχέση με παρόμοια user, card, product ή time contexts.

Ασφαλής διευκρίνιση:

> Το simulated user anchor είναι proxy. Είναι χρήσιμο για αυτό το benchmark, αλλά σε production ένα πραγματικό account identifier θα ήταν προτιμότερο.

### Διαφάνεια 11 - Experimental Configurations

Πες:

> Τα πειράματα οργανώνονται σε τέσσερις διαμορφώσεις. Πρώτα, το baseline χρησιμοποιεί όλα τα features και την αρχική ανισορροπία. Δεύτερον, εφαρμόζω downsampling στη majority class με ratio 1:5, μόνο στο training data. Τρίτον, μειώνω το feature set με cross-model SHAP agreement. Τέταρτον, μειώνω το decision threshold στο 0.1 για να μελετήσω το operational trade-off ανάμεσα σε recall και precision.

Κρίσιμη φράση:

> Το test set παραμένει πάντα untouched και imbalanced.

### Διαφάνεια 12 - Baseline Results

Πες:

> Στο baseline, το LightGBM πέτυχε το υψηλότερο ROC-AUC, 0.918, ακολουθούμενο από CatBoost με 0.910 και XGBoost με 0.896. Το LightGBM είχε επίσης το καλύτερο PR-AUC και F1 στο default threshold. Το CatBoost είχε υψηλότερο recall, αλλά πολύ χαμηλότερο precision, άρα έπιανε περισσότερα fraud cases με κόστος περισσότερα false positives.

Μετά πες:

> Η διπλωματική δίνει μία πιθανή εξήγηση για την απόδοση του LightGBM: το histogram-based split finding χειρίζεται αποτελεσματικά high-dimensional feature spaces και μπορεί να λειτουργήσει σαν ήπια regularization σε έντονη ανισορροπία. Το λέω προσεκτικά ως εξήγηση για αυτό το dataset και αυτό το protocol, όχι ως γενικό κανόνα ότι το LightGBM είναι πάντα καλύτερο.

### Διαφάνεια 13 - Downsampling

Πες:

> Το downsampling αλλάζει την training distribution ώστε η majority class να μην κυριαρχεί τόσο πολύ στη μάθηση. Το XGBoost και το CatBoost βελτίωσαν το ROC-AUC, ενώ το LightGBM έμεινε σχεδόν σταθερό σε ROC-AUC και βελτιώθηκε σε PR-AUC. Αυτό δείχνει ότι το downsampling βοήθησε τα μοντέλα να μάθουν καλύτερα τη minority class, αλλά δεν βελτιώνει αυτόματα κάθε threshold-dependent metric.

Φράση άμυνας:

> Το downsampling εφαρμόστηκε μόνο στο training data. Το test distribution παρέμεινε ρεαλιστικό.

### Διαφάνεια 14 - SHAP

Πες:

> Το SHAP χρησιμοποιήθηκε για να εντοπίσω features που συνεισφέρουν σταθερά στα μοντέλα. Το reduced set κράτησε features που βρίσκονταν στο top 30% για τουλάχιστον δύο από τα τρία μοντέλα. Έτσι ο χώρος εισόδου μειώθηκε από 748 engineered model inputs σε 215. Τα πιο σημαντικά features περιλαμβάνουν τόσο αρχικά transaction variables όσο και engineered behavioral aggregates, κάτι που στηρίζει τη χρησιμότητα του feature engineering.

Αν ρωτηθείς για causality:

> Το SHAP εξηγεί model attribution. Δεν αποδεικνύει αιτιότητα.

### Διαφάνεια 15 - Feature Reduction

Πες:

> Μετά τη μείωση των features, το ROC-AUC παρέμεινε πολύ κοντά στα downsampled full-feature results. Το LightGBM βελτιώθηκε ελαφρά σε 0.919, ενώ XGBoost και CatBoost παρέμειναν κοντά. Η ερμηνεία είναι ότι ο αρχικός expanded feature space είχε redundancy, και η consensus-based feature selection κράτησε το μεγαλύτερο μέρος του χρήσιμου σήματος.

Σημαντικό:

> Αυτό το πείραμα αφορά απλοποίηση του feature space χωρίς ουσιαστική απώλεια discrimination.

### Διαφάνεια 16 - Threshold Tuning

Πες:

> Το threshold experiment δείχνει την επιχειρησιακή πλευρά του fraud detection. Στο default threshold 0.5, το recall είναι μέτριο. Μειώνοντας το threshold στο 0.1, το recall αυξάνεται έντονα και τα μοντέλα φτάνουν κοντά ή πάνω από 0.9. Όμως το precision πέφτει σημαντικά, δηλαδή σημαίνονται πολύ περισσότερες legitimate συναλλαγές ως ύποπτες. Αυτό δεν είναι απλά model improvement. Είναι business trade-off.

Φράση άμυνας:

> Δεν ισχυρίζομαι ότι το 0.1 είναι το optimal production threshold. Το χρησιμοποιώ για να δείξω πώς το threshold ελέγχει recall και false positives.

### Διαφάνεια 17 - Champion Scorecard

Πες:

> Στις τρεις βασικές ROC-AUC διαμορφώσεις, το LightGBM είναι το πιο σταθερό μοντέλο, περίπου από 0.917 έως 0.919. Το XGBoost επωφελείται περισσότερο από το downsampling, ενώ το CatBoost πλησιάζει το LightGBM μετά το downsampling. Άρα το συμπέρασμα δεν είναι απλώς "κερδίζει το LightGBM", αλλά ότι sampling και feature design αλλάζουν τη συμπεριφορά των μοντέλων.

Κράτησέ το σύντομο.

### Διαφάνεια 18 - Synthesis

Πες:

> Η σύνθεση είναι ότι το fraud detection είναι pipeline problem. Το model choice έχει σημασία, αλλά εξίσου σημαντικά είναι το feature engineering, η διαχείριση της ανισορροπίας, η επιλογή features και το operating threshold. Το ROC-AUC δείχνει ranking ability, ενώ precision και recall δείχνουν τις επιχειρησιακές συνέπειες.

Κύρια πρόταση:

> Το καλύτερο model score δεν αρκεί. Για deployment χρειάζεται και σωστή επιλογή operating point.

### Διαφάνεια 19 - Conclusions

Πες:

> Τα βασικά συμπεράσματα είναι τέσσερα. Πρώτον, τα gradient boosting models είναι αποτελεσματικά για αυτό το structured fraud dataset. Δεύτερον, το LightGBM ήταν το πιο σταθερό μοντέλο ως προς ROC-AUC. Τρίτον, τα engineered behavioral features ήταν σημαντικά και βοήθησαν να διατηρηθεί η απόδοση μετά τη μείωση χαρακτηριστικών. Τέταρτον, το threshold tuning έχει μεγάλη επιχειρησιακή επίδραση και πρέπει να επιλέγεται με βάση κόστος και review capacity.

Μετά:

> Ως future work, τα πιο σημαντικά σημεία είναι cost-aware thresholds, probability calibration, πλουσιότερα behavioral ή relational data, και drift-aware evaluation.

Πες καθαρά τους περιορισμούς:

> Οι βασικοί περιορισμοί είναι ότι χρησιμοποιείται δημόσιο ανωνυμοποιημένο benchmark dataset, το user ID είναι simulated, και το live demo είναι ενδεικτικό και όχι production fraud system.

### Διαφάνεια 20 - Live Demo

Πες:

> Αυτό το σύντομο demo δείχνει πώς το τελικό LightGBM pipeline μπορεί να δώσει fraud probability και τοπική εξήγηση τύπου SHAP για μία συναλλαγή. Ο σκοπός δεν είναι να ισχυριστώ production deployment, αλλά να γίνει πιο κατανοητή η συμπεριφορά του μοντέλου: το score δείχνει risk, και η εξήγηση δείχνει ποια features έσπρωξαν την πρόβλεψη προς τα πάνω ή προς τα κάτω.

Κράτησε το demo στα 60-90 δευτερόλεπτα.

Αν αποτύχει το demo:

> Το demo είναι visualization layer πάνω στην ιδέα του trained-model pipeline. Τα αποτελέσματα της διπλωματικής βασίζονται στα offline πειράματα που παρουσιάστηκαν ήδη, όχι στο live demo περιβάλλον.

### Διαφάνεια 21 - Thank You

Πες:

> Σας ευχαριστώ πολύ για την προσοχή σας. Είμαι στη διάθεσή σας για ερωτήσεις.

Σταμάτα εκεί. Μην προσθέσεις άλλα.

## Κεφάλαια που Πρέπει να Ξαναδιαβάσεις

Διάβασέ τα μέχρι να μπορείς να τα εξηγήσεις χωρίς σημειώσεις:

- Κεφάλαιο 1: σκοπός, ερευνητικά ερωτήματα, συνεισφορά.
- Κεφάλαιο 3: chronological split, Optuna, time-series CV, μοντέλα, metrics.
- Section 4.1: class imbalance, missing values, TransactionDT, amount patterns, categorical patterns.
- Section 4.2: simulated user ID, aggregate features, frequency features.
- Section 5.1: baseline results, metric table, LightGBM reasoning.
- Section 5.2: downsampling, ratio 1:5, τι βελτιώθηκε και τι όχι.
- Section 5.3: SHAP/top 30% agreement, feature reduction 748 σε 215, αποτελέσματα.
- Section 5.4: threshold 0.1, αύξηση recall, πτώση precision, operational interpretation.
- Section 5.5: LightGBM stability, threshold impact μεγαλύτερο από μικρές διαφορές μοντέλων.
- Κεφάλαιο 6: τέσσερα συμπεράσματα και future work.

Αν έχεις μόνο μία ώρα, διάβασε Κεφάλαιο 3 και Sections 5.1-5.5.

## Πιθανές Ερωτήσεις Άμυνας και Απαντήσεις

### Γιατί επέλεξες gradient boosting models;

Απάντηση:

> Το dataset είναι structured tabular transaction data με πολλά numeric, categorical και engineered features. Τα gradient boosting models είναι πολύ ισχυρά σε τέτοια δεδομένα, μπορούν να μάθουν nonlinear interactions, και είναι πιο πρακτικά και πιο ερμηνεύσιμα από πιο πολύπλοκες λύσεις. Επιπλέον μπορούν να συνδυαστούν με feature importance και SHAP.

### Γιατί όχι deep learning;

Απάντηση:

> Το deep learning μπορεί να είναι πολύ χρήσιμο, ειδικά όταν υπάρχουν sequential ή relational δεδομένα. Όμως αυτή η διπλωματική εστιάζει σε structured tabular benchmark. Σε αυτό το πλαίσιο, τα gradient boosting models δίνουν πολύ καλή ισορροπία ανάμεσα σε απόδοση, πρακτικότητα και ερμηνευσιμότητα. Deep learning θα ήταν λογική επέκταση με πλουσιότερα sequential ή graph data.

### Γιατί πήγε καλύτερα το LightGBM;

Απάντηση:

> Σε αυτό το dataset και με αυτό το evaluation protocol, το LightGBM ήταν το πιο σταθερό σε ROC-AUC. Η διπλωματική το συνδέει με το histogram-based split finding, που χειρίζεται αποδοτικά high-dimensional feature spaces και μπορεί να λειτουργήσει σαν ήπια regularization σε έντονη ανισορροπία. Δεν το γενικεύω ότι το LightGBM είναι πάντα το καλύτερο για fraud detection.

### Γιατί ROC-AUC ως βασικό metric;

Απάντηση:

> Το ROC-AUC είναι threshold-independent και ευθυγραμμίζεται με το IEEE-CIS/Kaggle benchmark. Μετρά την ικανότητα ranking σε όλα τα thresholds. Επειδή όμως το fraud είναι πολύ imbalanced, αναφέρω και PR-AUC, precision, recall και F1 για να αποτυπώσω την επιχειρησιακή συμπεριφορά.

### Γιατί όχι accuracy;

Απάντηση:

> Το accuracy είναι παραπλανητικό, επειδή μόνο περίπου 3.5% των συναλλαγών είναι fraud. Ένα μοντέλο μπορεί να προβλέπει σχεδόν τα πάντα ως legitimate και να φαίνεται accurate, ενώ αποτυγχάνει στην πραγματική ανίχνευση απάτης.

### Γιατί chronological split;

Απάντηση:

> Η απάτη έχει χρονική διάσταση. Ένα random split μπορεί να επιτρέψει σε μελλοντικά patterns να επηρεάσουν το training και να δώσει υπεραισιόδοξα αποτελέσματα. Το chronological split προσομοιώνει καλύτερα την πραγματική χρήση: εκπαιδεύουμε σε παλιότερες συναλλαγές και προβλέπουμε νεότερες.

### Το feature engineering δημιούργησε leakage;

Απάντηση:

> Η διπλωματική αντιμετωπίζει το feature engineering από time-realistic perspective. Το evaluation είναι chronological και το model selection γίνεται μόνο μέσα στο training period. Σε production υλοποίηση, τα aggregate features πρέπει να υπολογίζονται μόνο από ιστορική πληροφορία διαθέσιμη πριν από τη συναλλαγή.

### Γιατί downsampling;

Απάντηση:

> Η αρχική κατανομή είναι έντονα imbalanced, οπότε η majority class μπορεί να κυριαρχεί στο learning. Το downsampling δίνει πιο καθαρό σήμα στη minority class. Εφαρμόστηκε μόνο στο training data, ενώ το test set παρέμεινε στην αρχική imbalanced κατανομή.

### Γιατί ratio 1:5;

Απάντηση:

> Το 1:5 χρησιμοποιείται ως πρακτικός συμβιβασμός. Μειώνει την κυριαρχία της majority class χωρίς να κάνει την training distribution υπερβολικά τεχνητή. Πιο επιθετικά ratios θα μπορούσαν να εξεταστούν σε μελλοντική εργασία.

### Γιατί όχι SMOTE;

Απάντηση:

> Το SMOTE δημιουργεί synthetic minority examples. Στο fraud detection, synthetic fraud transactions μπορεί να εισάγουν μη ρεαλιστικά patterns. Για αυτό η διπλωματική προτίμησε το downsampling ως απλούστερη και πιο ασφαλή στρατηγική imbalance handling.

### Τι αποδεικνύει το SHAP;

Απάντηση:

> Το SHAP εξηγεί πώς το trained model αποδίδει σημασία στα features για τις προβλέψεις. Δεν αποδεικνύει αιτιότητα. Στη διπλωματική χρησιμοποιείται για interpretability και feature selection.

### Γιατί μείωση χαρακτηριστικών;

Απάντηση:

> Το feature engineering δημιούργησε μεγάλο feature space. Η μείωση χαρακτηριστικών εξετάζει αν μπορεί να διατηρηθεί η discriminatory power με λιγότερα και πιο σταθερά σημαντικά features. Το αποτέλεσμα ήταν ότι η απόδοση παρέμεινε πολύ κοντά, με χαμηλότερη πολυπλοκότητα.

### Είναι το threshold 0.1 προτεινόμενο για παραγωγή;

Απάντηση:

> Όχι. Το threshold 0.1 είναι πειραματικό operating point για να φανεί το recall-precision trade-off. Σε production, το threshold πρέπει να επιλεγεί με βάση fraud-loss cost, false-positive cost, review capacity και calibration.

### Ποιο είναι το πιο σημαντικό αποτέλεσμα;

Απάντηση:

> Το πιο σημαντικό αποτέλεσμα είναι ότι έχει σημασία ολόκληρο το fraud-detection pipeline. Το LightGBM ήταν το πιο σταθερό σε ROC-AUC, αλλά sampling, feature selection και threshold choice επηρέασαν έντονα την επιχειρησιακή συμπεριφορά.

### Ποιοι είναι οι βασικοί περιορισμοί;

Απάντηση:

> Οι βασικοί περιορισμοί είναι το ανωνυμοποιημένο δημόσιο dataset, τα simulated user identifiers, η έλλειψη πραγματικών business cost labels, η απουσία production calibration και η απουσία μακροχρόνιας drift evaluation.

### Πώς θα βελτίωνες την εργασία;

Απάντηση:

> Θα πρόσθετα cost-aware threshold optimization, probability calibration, πιο πλούσια temporal και relational features, περισσότερα και πιο διαφορετικά transaction data, και drift-aware evaluation.

## Κανόνες για Μείωση Άγχους

- Μίλα με δομή claim και evidence: "Ο ισχυρισμός είναι Χ, και το evidence είναι Υ."
- Χρησιμοποίησε τη φράση "σε αυτό το dataset και με αυτό το protocol" όταν συγκρίνεις μοντέλα.
- Αν δεν είσαι σίγουρος, στένεψε τον ισχυρισμό αντί να τον υπερασπιστείς υπερβολικά.
- Μετά από κάθε results slide, κάνε μικρή παύση πριν δώσεις interpretation.
- Μην απολογείσαι για περιορισμούς. Παρουσίασέ τους ως scope boundaries και future work.
- Αν σε πιέσουν, γύρνα στο experimental protocol: chronological split, untouched test set, metrics κατάλληλα για imbalance.

## Τρεις Προτάσεις για Αποστήθιση

1. > Η διπλωματική δεν αφορά μόνο την επιλογή του καλύτερου μοντέλου. Μελετά πώς συμπεριφέρεται ολόκληρο το fraud-detection pipeline υπό class imbalance, feature reduction και διαφορετικά operating thresholds.
2. > Το LightGBM ήταν το πιο σταθερό μοντέλο σε ROC-AUC σε αυτό το dataset και με αυτό το protocol, αλλά οι deployment αποφάσεις εξαρτώνται από threshold, κόστος και review capacity.
3. > Το threshold experiment δείχνει ότι το fraud detection είναι επιχειρησιακό trade-off: μπορούμε να αυξήσουμε το recall, αλλά αυτό φέρνει περισσότερα false positives.
